import asyncio
import logging
from pathlib import Path
from typing import Iterable, Sequence
from uuid import UUID

from supriya.contexts import AsyncServer, Buffer

from ..core.milvus import Entry
from .api_client import APIClient

logger = logging.getLogger(__name__)


class BufferManager:
    def __init__(
        self, api_client: APIClient, context: AsyncServer, download_directory: Path
    ) -> None:
        self.api_client = api_client
        self.context = context
        self.download_directory = download_directory
        self.entry_futures: dict[str, asyncio.Future] = {}
        self.reference_to_buffer_ids: dict[UUID | int, set[int]] = {}
        self.buffer_id_to_entry: dict[int, str] = {}
        self.buffer_id_to_references: dict[int, set[UUID | int]] = {}

    def entry_to_key(self, entry: Entry) -> str:
        return f"{entry['digest']}-{entry['start_frame']}-{entry['frame_count']}"

    async def acquire(
        self, reference: UUID, entries: Iterable[Entry]
    ) -> Sequence[Buffer]:
        """
        Acquire multiple buffer references.

        Download from the API if necessary, or wait on ongoing downloads.
        """
        buffers = await asyncio.gather(
            *[self.acquire_one(reference=reference, entry=entry) for entry in entries]
        )
        return sorted(buffers, key=lambda x: x.id_)

    async def acquire_one(self, reference: UUID, entry: Entry) -> Buffer:
        """
        Acquire single buffer reference.

        Download from the API if necessary, or wait on ongoing downloads.
        """
        logger.debug(f"Acquiring: {reference} {entry}")
        key = self.entry_to_key(entry)
        if key not in self.entry_futures:
            # Create a future other acquire tasks can wait on
            self.entry_futures[key] = asyncio.get_running_loop().create_future()
            # Fetch the audio from the API
            data = await self.api_client.audio_fetch(
                digest=entry["digest"],
                start_frame=entry["start_frame"],
                frame_count=entry["frame_count"],
            )
            # Write the audio data to disk
            (path := self.download_directory / f"{key}.wav").write_bytes(data)
            logger.debug(f"Fetched {entry} to {path}")
            # Allocate a buffer
            buffer = self.context.add_buffer(file_path=path)
            # Sync the server so we know it's available
            await self.context.sync()
            # Delete the audio file
            path.unlink()
            # Set the future result
            self.buffer_id_to_entry[buffer.id_] = key  # affords cleanup later
            self.entry_futures[key].set_result(buffer)
        # Wait for the future result (no-op if this same function downloaded it)
        buffer = await self.entry_futures[key]
        # Increment references
        self.increment(reference, buffer.id_)
        logger.debug(f"Acquired: {reference} {entry}")
        return buffer

    def increment(self, reference: UUID | int, buffer_id: int) -> None:
        """
        Increment references to a buffer.
        """
        logger.debug(f"Incrementing: {reference} for {buffer_id}")
        # print("INCREMENT", "REF", reference, "BUF", buffer_id)
        self.buffer_id_to_references.setdefault(buffer_id, set()).add(reference)
        self.reference_to_buffer_ids.setdefault(reference, set()).add(buffer_id)

    def decrement(self, reference: UUID | int) -> None:
        """
        Decrement references to a buffer.

        Release if reference count drops to zero.
        """
        # The node ID may not correspond to a node using a buffer
        for buffer_id in sorted(self.reference_to_buffer_ids.pop(reference, [])):
            logger.debug(f"Decrementing: {reference} for {buffer_id}")
            # print("DECREMENT", "REF", reference, "BUF", buffer_id)
            self.buffer_id_to_references[buffer_id].remove(reference)
            if self.buffer_id_to_references[buffer_id]:
                continue
            logger.debug(f"Releasing: {buffer_id}")
            # print("RELEASING", "BUF", buffer_id)
            self.buffer_id_to_references.pop(buffer_id)
            self.entry_futures.pop(
                self.buffer_id_to_entry.pop(buffer_id)
            ).result().free()
