import configparser
from utils import static_init


@static_init
class GlobalConfig:
    default_file_name = "config.ini"
    config = configparser.ConfigParser()

    @classmethod
    def get_section(cls, section_name):
        if section_name in cls.config:
            return cls.config[section_name].keys()
        else:
            return None

    @classmethod
    def get(cls, section_name, attr_name):
        if section_name in cls.config and attr_name in cls.config[section_name]:
            value = cls.config.get(section_name, attr_name)
            value = value.split(":")
            type_name = value[0]
            value = ":".join(value[1:])
            if type_name == "str":
                value = str(value)
            elif type_name == "float":
                value = float(value)
            elif type_name == "int":
                value = int(value)
            elif type_name == "bool":
                value = bool(value)

            return value
        else:
            return None

    @classmethod
    def __static_init__(cls):
        cls.config.read(cls.default_file_name)
