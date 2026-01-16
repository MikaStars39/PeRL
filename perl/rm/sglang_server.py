import asyncio
import logging
import os
import signal
import subprocess
import sys

import aiohttp

logger = logging.getLogger("AllInOne-RM")


class SGLangManager:
    """Lifecycle manager for the SGLang subprocess."""

    def __init__(self, config):
        self.config = config
        self.process = None

    def start(self):
        """Start the SGLang subprocess."""
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.config.model_path,
            "--port", str(self.config.sglang_port),
            "--host", self.config.sglang_host,
            "--trust-remote-code",
        ]

        # Optional parallelism settings.
        if getattr(self.config, "sglang_tp_size", 1) > 1:
            cmd.extend(["--tp-size", str(self.config.sglang_tp_size)])
        if getattr(self.config, "sglang_dp_size", 1) > 1:
            cmd.extend(["--dp-size", str(self.config.sglang_dp_size)])
        
        logger.info(f"🚀 Starting SGLang service (Model: {self.config.model_path})...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Start a new process group so we can terminate it safely.
        self.process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            preexec_fn=os.setsid,
        )

    async def wait_until_ready(self):
        """Poll the health endpoint until SGLang is ready."""
        health_url = f"http://localhost:{self.config.sglang_port}/health"
        logger.info("⏳ Waiting for model warmup (may take a few minutes)...")
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(health_url) as resp:
                        if resp.status == 200:
                            logger.info("✅ SGLang service is ready.")
                            return
                except Exception:
                    pass
                
                # Exit early if the subprocess died.
                if self.process.poll() is not None:
                    logger.error("❌ SGLang exited unexpectedly. Check VRAM or model path.")
                    sys.exit(1)
                
                await asyncio.sleep(5)

    def stop(self):
        """Stop the SGLang subprocess safely."""
        if self.process:
            logger.info("🛑 Stopping SGLang service...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception as e:
                logger.warning(f"Stop process issue (may already be closed): {e}")
            logger.info("👋 SGLang service stopped.")
