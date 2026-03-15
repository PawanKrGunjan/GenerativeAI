from chat.chat_run import handle_user_message
from chat.session_manager import SESSION_MANAGER


async def run_chat(session_id: str, message: str):

    thread_id = SESSION_MANAGER.get_thread(session_id)

    result = await handle_user_message(
        thread_id=thread_id,
        user_query=message
    )

    return result