"""Tests for ChatHistoryStore."""

import pytest
import pytest_asyncio

from personal_kb.web.chat_history import ChatHistoryStore, derive_title


@pytest_asyncio.fixture
async def store():
    s = await ChatHistoryStore.open(db_path=":memory:")
    yield s
    await s.close()


class TestDeriveTitle:
    def test_short_text_unchanged(self):
        assert derive_title("Hello world") == "Hello world"

    def test_long_text_truncated(self):
        text = "A" * 100
        result = derive_title(text)
        assert len(result) <= 83  # 80 + "..."
        assert result.endswith("...")

    def test_newlines_replaced(self):
        assert derive_title("line one\nline two") == "line one line two"

    def test_empty_string(self):
        assert derive_title("") == ""

    def test_whitespace_stripped(self):
        assert derive_title("  hello  ") == "hello"


class TestChatHistoryStore:
    @pytest.mark.asyncio
    async def test_create_and_list(self, store):
        await store.create_chat("c1", "First chat", "summarize")
        await store.create_chat("c2", "Second chat", "explore")

        chats = await store.list_chats()
        assert len(chats) == 2
        # Most recent first
        assert chats[0]["id"] == "c2"
        assert chats[1]["id"] == "c1"
        assert chats[0]["mode"] == "explore"

    @pytest.mark.asyncio
    async def test_save_and_get_messages(self, store):
        await store.create_chat("c1", "Test", "")
        await store.save_message("c1", "user", "Hello")
        await store.save_message("c1", "assistant", "Hi there")
        await store.save_message("c1", "user", "Follow up")
        await store.save_message("c1", "assistant", "Sure thing")

        msgs = await store.get_messages("c1")
        assert len(msgs) == 4
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there"}
        assert msgs[2] == {"role": "user", "content": "Follow up"}
        assert msgs[3] == {"role": "assistant", "content": "Sure thing"}

    @pytest.mark.asyncio
    async def test_save_messages_bulk(self, store):
        await store.create_chat("c1", "Bulk test", "")
        await store.save_messages_bulk(
            "c1",
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
            ],
        )

        msgs = await store.get_messages("c1")
        assert len(msgs) == 3
        assert msgs[0]["content"] == "Q1"

    @pytest.mark.asyncio
    async def test_delete_cascades(self, store):
        await store.create_chat("c1", "To delete", "")
        await store.save_message("c1", "user", "Hello")
        await store.save_message("c1", "assistant", "Hi")

        deleted = await store.delete_chat("c1")
        assert deleted is True
        assert await store.get_messages("c1") == []
        assert await store.chat_exists("c1") is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        assert await store.delete_chat("bogus") is False

    @pytest.mark.asyncio
    async def test_chat_exists(self, store):
        assert await store.chat_exists("c1") is False
        await store.create_chat("c1", "Exists", "")
        assert await store.chat_exists("c1") is True

    @pytest.mark.asyncio
    async def test_list_respects_limit(self, store):
        for i in range(5):
            await store.create_chat(f"c{i}", f"Chat {i}", "")
        chats = await store.list_chats(limit=3)
        assert len(chats) == 3

    @pytest.mark.asyncio
    async def test_save_message_updates_chat_timestamp(self, store):
        await store.create_chat("c1", "Test", "")
        chats_before = await store.list_chats()
        ts_before = chats_before[0]["updated_at"]

        await store.save_message("c1", "user", "New message")
        chats_after = await store.list_chats()
        ts_after = chats_after[0]["updated_at"]

        assert ts_after >= ts_before
