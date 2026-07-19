"""Responsive execution adapter for engines with blocking generation steps."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from .engine import EngineEvent, EngineProtocol, GenerationRequest, ModelMetadata


_WORKER_DONE = object()


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    error: BaseException


class EngineRunner:
    """Serialize real-engine generations and keep their blocking work off-loop."""

    def __init__(self, engine: EngineProtocol) -> None:
        self._engine = engine
        self._generation_lock = asyncio.Lock()
        self._accepting = True
        self._active_generations = 0
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._engine.model_metadata

    @property
    def ready(self) -> bool:
        return self._accepting

    async def generate(self, request: GenerationRequest):
        if not self._accepting:
            raise RuntimeError("engine runner is shutting down")
        self._active_generations += 1
        self._drained.clear()
        try:
            if self.model_metadata.engine != "treekv":
                async for event in self._engine.generate(request):
                    yield event
                return

            async with self._generation_lock:
                async for event in self._generate_in_worker(request):
                    yield event
        finally:
            self._active_generations -= 1
            if self._active_generations == 0:
                self._drained.set()

    async def shutdown(self) -> None:
        """Stop admission and wait for every admitted generation to finish."""
        self._accepting = False
        await self._drained.wait()

    async def _generate_in_worker(self, request: GenerationRequest):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[EngineEvent | _WorkerFailure | object] = asyncio.Queue()

        def publish(item: EngineEvent | _WorkerFailure | object) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                # The interpreter/event loop is already being force-closed.
                pass

        async def consume() -> None:
            async for event in self._engine.generate(request):
                publish(event)

        def worker() -> None:
            try:
                asyncio.run(consume())
            except BaseException as error:
                publish(_WorkerFailure(error))
            finally:
                publish(_WORKER_DONE)

        thread = threading.Thread(
            target=worker,
            name=f"autotree-{self.model_metadata.engine}-generation",
            daemon=False,
        )
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is _WORKER_DONE:
                    break
                if isinstance(item, _WorkerFailure):
                    raise item.error
                yield item
        finally:
            if thread.is_alive():
                await asyncio.shield(asyncio.to_thread(thread.join))


__all__ = ["EngineRunner"]
