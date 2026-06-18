#chat/chat_service.py
import logging
from chat.chat_run import handle_user_message
from chat.session_manager import SESSION_MANAGER

LOGGER = logging.getLogger("chat_run")

async def run_chat(session_id: str, message: str):

    thread_id = SESSION_MANAGER.get_thread(session_id)

    result = await handle_user_message(
        thread_id=thread_id,
        user_query=message
    )

    LOGGER.info(f"RESULT TYPE: {type(result)}")
    LOGGER.info(f"RESULT: {result}")

    return result