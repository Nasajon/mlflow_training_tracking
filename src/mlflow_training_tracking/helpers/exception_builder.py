
class ExceptionBuilder:
    def __init__(self, exception, separator):
        self.messages = list()
        self.separator = separator
        self.exception = exception

    def add_message(self, message):
        self.messages.append(message)

    def set_messages(self, messages):
        self.messages = messages

    def raise_exception_if_exist(self, exception_class=Exception):
        if len(self.messages) == 0:
            return
        messages = self.separator.join(self.messages)
        raise self.exception(messages)