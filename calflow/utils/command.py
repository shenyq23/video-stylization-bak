import os
import time
from dataclasses import dataclass
from typing import Optional

DISPLAY_PREFIX = ["section", "option", "full option"]

@dataclass
class CommandSection:
    dash_cnt: int
    
    # important flags will be printed in the command output
    is_important: bool
    key: str
    value: Optional[str]
    
    def command_str(self):
        if self.value is None:
            return "{}{}".format("-" * self.dash_cnt, self.key)
        else:
            return "{}{} {}".format("-" * self.dash_cnt, self.key, self.value)
        
    def display_str(self):
        if self.value is None:
            return "[{}] {}".format(DISPLAY_PREFIX[self.dash_cnt], self.key)
        else:
            return "[{}] {}: {}".format(DISPLAY_PREFIX[self.dash_cnt], self.key, self.value)

class Command:
    def __init__(self, cmd_base):
        self.cmd_base = cmd_base
        self.output_path = None
        self.sections = []
        
    def add_section(self, section, is_important=False):
        self.sections.append(CommandSection(
            dash_cnt=0,
            is_important=is_important,
            key=section,
            value=None,
        ))
        return self

    def add_flag(self, key, value=None, is_full=False, is_important=False):
        self.sections.append(CommandSection(
            dash_cnt=2 if is_full else 1,
            is_important=is_important,
            key=key,
            value=value,
        ))
        return self
    
    def set_output_path(self, output_path):
        # @note: idk why but it works
        self.output_path = output_path.replace("\\", "/")
        return self
    
    def command_str(self):
        cmd = self.cmd_base
        for section in self.sections:
            cmd += " {}".format(section.command_str())
        if self.output_path:
            cmd += " > {}".format(self.output_path)
        return cmd
    
    def display_str(self):
        display = "[executing] {}".format(self.cmd_base)
        for section in self.sections:
            if section.is_important:
                display += "\n" + section.display_str()
        if self.output_path:
            display += "\n[output path] {}".format(self.output_path)
        return display + "\n"
    
    def tmux_dispatch_command_str(self, session_name):
        cmd = self.command_str()
        return "tmux new-session -d -s {} '{}'".format(session_name, cmd)
        
    def run(self):
        cmd = self.command_str()
        print("[exec time] {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        print(self.display_str())

        if self.output_path is None:
            os.system(cmd)
        else:
            os.system(cmd + "> {}".format(self.output_path))

if __name__ == "__main__":
    cmd = Command("ffmpeg")
    cmd.add_section("input", is_important=True).add_flag("i", "input.mp4", is_full=True, is_important=True)
    cmd.add_section("output", is_important=True).add_flag("c:v", "libx264").add_flag("b:v", "1000k")
    cmd.set_output_path("output.mp4")
    
    print(cmd.command_str())
    print(cmd.display_str())
    
    # cmd.run()
    print(cmd.tmux_dispatch_command_str("test_session"))
    