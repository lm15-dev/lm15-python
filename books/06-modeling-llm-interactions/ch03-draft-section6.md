## The Consequence Problem

Everything in this book up to the word "tool" was reversible. A bad message
representation can be refactored. A bad conversation model can be redesigned. A
bad streaming abstraction can be replaced. These are software decisions,
contained within the codebase, undone by changing the code.

Tool execution introduces a category of outcome that code changes can't undo. A
file written to disk persists after the program exits. An email sent to a
customer exists in their inbox. A database row deleted is gone (unless you have
backups, and you restored them in time, and the deletion was noticed before
dependent systems propagated the absence). A payment processed moves real money.
The model's output has crossed from the symbolic world — where everything is
text, and text can be discarded — into the physical world, where actions have
consequences and consequences have consequences.

This changes what it means for the model to be wrong.

When the model generates incorrect text, the incorrectness is contained. The
user reads it, recognizes the error (or doesn't), and the worst case is
misinformation — real, but bounded. When the model generates an incorrect tool
call that executes, the error *compounds*. The wrong file is written. The next
turn reads that wrong file. The model incorporates the wrong content into its
reasoning. It makes further decisions based on corrupted state. By turn 15, the
codebase has been modified in ways that are internally consistent with the
original error but wrong in ways the model can't detect, because the model's
understanding of the codebase is built on the corrupt foundation of turn 3.

This is the cascading error problem, and it's specific to tool-using systems. In
text-only interactions, errors are local — each response is generated from the
conversation history, and a wrong response doesn't corrupt anything except the
next response's context (which can be fixed by clearing history). In tool-using
interactions, errors are global — a wrong tool execution modifies state that
persists outside the conversation, and subsequent tool calls operate on that
modified state. The conversation can be cleared. The filesystem can't.

### Two Kinds of Tools

The consequence problem doesn't apply uniformly. Some tools are read-only —
`read_file`, `search`, `list_directory`, `calculate`. They observe the world but
don't change it. The worst case is a wasted API call or irrelevant search
results. Calling them incorrectly costs tokens but does no damage.

Other tools are write-capable — `write_file`, `run_command`, `send_message`,
`execute_sql`, `deploy`. They change the world. The worst case is unbounded.

This distinction — observation vs action, query vs mutation, GET vs POST — is
fundamental to the consequence problem, and it maps directly onto the control
spectrum from the previous section. Read-only tools are safe to auto-execute.
Write-capable tools need gates. The hybrid pattern (auto-execute reads, approve
writes) is a direct application of this distinction.

But the distinction isn't always clean. `run_command("ls -la")` is a read.
`run_command("rm -rf /")` is catastrophic. Same tool, same schema, same
auto-execution policy — different consequences. The tool's risk profile depends
on its arguments, not just its identity. A truly safe system would inspect
arguments before execution, not just tool names. But argument inspection
requires understanding what arguments are dangerous, which is
application-specific and difficult to generalize. `rm` is dangerous. `grep` is
safe. `curl` could be either. The library can't draw this line.

### The Responsibility Question

When a model auto-executes a tool call that causes damage, who is responsible?

This is not a philosophical question. It's a practical one that determines where
safeguards are built and who pays when they fail.

**The model trainer** selected the training data, designed the reward function,
and shaped the model's tendency to generate tool calls. The model that calls `rm
-rf` learned something — from training data, from RLHF, from the patterns it
absorbed — that made that call seem like a reasonable next step. But the model
trainer can't anticipate every tool, every schema, every deployment context. The
trainer's responsibility ends at "produce a model that follows instructions and
uses tools coherently." The specific tools and their consequences are
downstream.

**The library author** decided that callables auto-execute. The library provided
the machinery that translated the model's token output into a Python function
call and ran it. lm15's `Model.__call__` has a loop that executes callables
without any safety check beyond matching the function name. The library author's
defense: the developer chose to pass the callable, knowing (or having the
documentation to know) that it would auto-execute. The library provided the
mechanism; the developer provided the intent.

**The application developer** chose which tools to expose, chose the execution
mode (auto vs manual), chose whether to add argument validation, and chose to
run the system unsupervised. The developer is the only actor in this chain who
knows the deployment context — whether `write_file` is writing to a sandbox or
to production, whether `send_email` is going to a test inbox or to real users.

The practical answer is that responsibility follows knowledge. The model doesn't
know what tools are dangerous. The library doesn't know the deployment context.
The developer knows both. The safeguards belong in the developer's code —
argument validation, approval gates, sandboxing, audit logging — because the
developer is the only one with enough information to build them correctly.

But this answer has an uncomfortable corollary: the developer who skips the
safeguards — who passes `run_command` as a callable to an auto-executing agent
running in production — bears the responsibility for what happens. The library
let them. The model made the call. But the developer set the stage.

### What Libraries Can Do

A library can't solve the consequence problem — the problem is in the world, not
in the code. But a library can make the developer's job easier.

**Default to safety.** If the default execution mode is manual, the developer
must opt in to auto-execution explicitly. This is inconvenient for simple tools
and important for dangerous ones. The friction is a feature.

**Make execution visible.** Log every tool call — name, arguments, result — so
the developer can audit what the agent did. lm15's `model.history` contains this
information, but it's retrospective, not real-time. Streaming events
(`tool_call`, `tool_result`) provide real-time visibility.

**Expose the loop.** Don't hide the agent loop behind an abstraction. Let the
developer see every iteration, inject logic at every step, and terminate at any
point. The developer who can see the loop can control the loop. The developer
who can't is trusting the library with consequences the library doesn't
understand.

**Don't pretend to solve what you can't.** A library that offers "safe tool
execution" is making a promise it can't keep, because safety depends on context
the library doesn't have. A library that offers "tool execution with explicit
developer control" is making a promise it can keep.

The LLM community is still early in thinking about the consequence problem. The
current state of the art is the hybrid pattern — auto-execute reads, gate writes
— which is crude but effective, like a lock on a door. It doesn't prevent a
determined attacker (a model that accomplishes destructive goals through a
sequence of individually harmless read operations). It prevents accidents, which
is where most damage comes from.

The gap between what models can do — call any tool, with any arguments, in any
sequence — and what they should do is the central unsolved problem of agent
design. This book can't solve it. But it can name it clearly, which is the first
step toward designs that take it seriously.
