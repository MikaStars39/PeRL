#!/bin/bash

# --- Configuration ---
# Support 'kt' alias if present
shopt -s expand_aliases
if command -v kt >/dev/null 2>&1; then
    alias kubectl='kt'
fi

NAMESPACE="explore-train"
START_IDX=${1:-0}
END_IDX=${2:-0}
JOB_KEYWORDS=($3) # Supports multiple keywords if passed as a string/array

declare -A JOB_SUMMARY

# --- Helper Functions ---
log_info()  { echo -e "\033[0;34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
log_warn()  { echo -e "\033[0;33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }

clean_pod_processes() {
    local pod=$1
    log_info "Cleaning stale processes on $pod..."
    kubectl exec -n "$NAMESPACE" "$pod" -- pkill -f "sglang|ray|Megatron-LM" >/dev/null 2>&1 || true
}

# --- Validation ---
if [ -z "$3" ]; then
    log_error "Usage: $0 <start_idx> <end_idx> <job_keyword>"
fi

log_info "Scanning Ray Clusters in Namespace: $NAMESPACE..."

# Fetch all running pods once to minimize API calls
ALL_PODS_RAW=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers -o custom-columns=":metadata.name")

if [ -z "$ALL_PODS_RAW" ]; then
    log_error "No running pods found in namespace: $NAMESPACE"
fi

# --- Main Logic ---
for KEYWORD in "${JOB_KEYWORDS[@]}"; do
    # Filter pods matching the keyword and sort them
    CURRENT_JOB_PODS=($(echo "$ALL_PODS_RAW" | grep "^$KEYWORD" | sort -V))
    POD_COUNT=${#CURRENT_JOB_PODS[@]}

    echo "--------------------------------------------------------"
    log_info "Processing Job [$KEYWORD] | Nodes Found: $POD_COUNT"

    if [ "$START_IDX" -ge "$POD_COUNT" ]; then
        log_warn "START_IDX ($START_IDX) out of range for $KEYWORD. Skipping."
        continue
    fi

    # Identify Master and Worker Range
    MASTER_POD=${CURRENT_JOB_PODS[$START_IDX]}
    TOTAL_IDX=$((POD_COUNT - 1))
    
    # Adjust END_IDX if default (0) or out of bounds
    [[ $END_IDX -le 0 || $END_IDX -gt $TOTAL_IDX ]] && ACTUAL_END=$TOTAL_IDX || ACTUAL_END=$END_IDX
    WORKER_START=$((START_IDX + 1))

    # 1. Clean and Start Head Node (Master)
    clean_pod_processes "$MASTER_POD"
    kubectl exec -n "$NAMESPACE" "$MASTER_POD" -- ray stop --force >/dev/null 2>&1
    kubectl exec -n "$NAMESPACE" "$MASTER_POD" -- ray start --head --port=6379 >/dev/null 2>&1
    
    MASTER_IP=$(kubectl get pod -n "$NAMESPACE" "$MASTER_POD" -o jsonpath='{.status.podIP}')
    
    if [ -z "$MASTER_IP" ]; then
        log_error "Failed to retrieve IP for $MASTER_POD"
        continue
    fi

    JOB_SUMMARY["$KEYWORD"]="$MASTER_POD | $MASTER_IP"
    log_success "Master Ready: $MASTER_POD ($MASTER_IP)"

    # 2. Start Worker Nodes in Parallel
    if [ "$WORKER_START" -le "$ACTUAL_END" ]; then
        log_info "Joining Workers ${WORKER_START} to ${ACTUAL_END}..."
        for (( i=WORKER_START; i<=ACTUAL_END; i++ )); do
            WORKER_POD=${CURRENT_JOB_PODS[$i]}
            (
                clean_pod_processes "$WORKER_POD"
                kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- ray stop --force >/dev/null 2>&1
                kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- ray start --address="$MASTER_IP:6379" >/dev/null 2>&1
            ) &
        done
        wait
    else
        log_warn "No workers in range to join."
    fi

    log_success "Job [$KEYWORD] setup complete."
done

# --- Final Summary Table ---
echo -e "\n\033[1;36m================ RAY JOBS CLUSTER SUMMARY ================\033[0m"
printf "%-25s %-35s %-15s\n" "JOB_NAME" "MASTER_POD" "MASTER_IP"
echo "---------------------------------------------------------------------------"
for KEY in $(echo "${!JOB_SUMMARY[@]}" | tr ' ' '\n' | sort); do
    IFS='|' read -r M_NAME M_IP <<< "${JOB_SUMMARY[$KEY]}"
    printf "%-25s %-35s %-15s\n" "$KEY" "$M_NAME" "$M_IP"
done
echo -e "\033[1;36m==========================================================\033[0m"