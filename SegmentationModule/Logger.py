import logging
import verboselogs

class LoggerHandler(object):
    def __init__(self, settings):
        self.settings = settings

        if hasattr(self.settings, 'logger_name'):
            self.logger_name = self.settings.logger_name
        else:
            self.logger_name = 'FileLogger'

        self.logger = verboselogs.VerboseLogger(self.logger_name)
        self.logger.setLevel(logging.SPAM)  # Set the basic logger level
        self.file_handler = None

        if hasattr(self.settings, 'log_level_file'):
            self.log_level_file = self.settings.log_level_file
        else:
            self.log_level_file = logging.SPAM
        if hasattr(self.settings, 'log_level_stream'):
            self.log_level_stream = self.settings.log_level_stream
        else:
            self.log_level_stream = logging.SPAM
        if hasattr(self.settings, 'output_logs'):
            self.output_logs = self.settings.output_logs
        else:
            self.output_logs = ''

    def start(self):
        self.file_handler = logging.FileHandler(filename=self.output_logs + '_' + self.logger_name + ".log", mode='w')
        self.file_handler.setLevel(self.log_level_file)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s %(levelname)s - %(message)s'))
        self.logger.addHandler(self.file_handler)

        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(self.log_level_stream)
        self.stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s %(levelname)s - %(message)s'))
        self.logger.addHandler(self.stream_handler)

    def end(self):
        self.logger.removeHandler(self.file_handler)
        self.file_handler = None
        self.logger.removeHandler(self.stream_handler)
        self.stream_handler = None
