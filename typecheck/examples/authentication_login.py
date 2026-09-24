from lm15.login import Auth, TerminalUI

auth = Auth.local()
auth.login("xai", "device", ui=TerminalUI())
