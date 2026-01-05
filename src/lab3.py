import sys
#import time

#Scanner Class
class Scanner:
    OPCODES = {"load","loadI","store","add","sub","mult","lshift","rshift","output","nop"}
    
    def __init__(self, filename):
        self.filename = filename
        self.line_num = 1
        self.pos = 0
        self.current_line = ""

    def scan(self):
        tokens = []
        try:
            with open(self.filename, "r") as f:
                for line in f:
                    self.current_line = line.rstrip("\n\r")
                    self.pos = 0
                    while self.pos < len(self.current_line):
                        ch = self.current_line[self.pos]
                        if ch in " \t":
                            self.pos += 1
                            continue
                        elif ch == "/" and self._peek() == "/":
                            # comment token
                            tokens.append(("COMMENT", self.current_line[self.pos:], self.line_num))
                            break
                        elif ch == ",":
                            tokens.append(("COMMA", ",", self.line_num))
                            self.pos += 1
                        elif ch == "=" and self._peek() == ">":
                            tokens.append(("ARROW", "=>", self.line_num))
                            self.pos += 2
                        elif ch.isdigit():
                            tokens.append(self._scan_const())
                        elif ch == "r":
                            #Need to check for rshift
                            s = self.current_line
                            if s[self.pos:self.pos+6] == "rshift":
                                tokens.append(self._scan_opcode())
                            else:
                                tokens.append(self._scan_register())
                        elif ch.isalpha():
                            tokens.append(self._scan_opcode())
                        else:
                            tokens.append(("ERROR_TOKEN", ch, self.line_num))
                            self.pos += 1
                    tokens.append(("EOL", "\\n", self.line_num))
                    self.line_num += 1                   
        except FileNotFoundError:
            import sys
            print(f"ERROR: file '{self.filename}' not found", file=sys.stderr)
            exit(1)
        return tokens


    def _scan_const(self):
        start = self.pos
        s = self.current_line
        while self.pos < len(s) and s[self.pos].isdigit():
            self.pos += 1
        lex = s[start:self.pos]
        return ("CONST", lex, self.line_num)

    def _scan_register(self):
        start = self.pos
        self.pos += 1
        s = self.current_line
        while self.pos < len(s) and s[self.pos].isalnum():
            self.pos += 1
        #Include characters like "-" in token
        while self.pos < len(s) and s[self.pos] not in " \t,=>/":
            self.pos += 1
        lex = s[start:self.pos]
        if len(lex) < 2 or not lex[1:].isdigit():
            return ("ERROR_TOKEN", lex, self.line_num)
        return ("REGISTER", lex, self.line_num)

    def _scan_opcode(self):
        start = self.pos
        s = self.current_line
        while self.pos < len(s) and s[self.pos].isalpha():
            self.pos += 1
        lex = s[start:self.pos]
        if lex in self.OPCODES:
            return ("OPCODE", lex, self.line_num)
        else:
            return ("ERROR_TOKEN", lex, self.line_num)
    
    
    def _peek(self):
        s = self.current_line
        if self.pos + 1 < len(s):
            return s[self.pos + 1]
        return ""

class ILOperation:
    def __init__(self, line, opcode, op1=None, op2=None, op3=None):
        self.line = line
        self.opcode = opcode
        # up to three integer operand fields (register numbers or constants)
        self.op1 = op1
        self.op2 = op2
        self.op3 = op3
        # placeholders
        self.VR = None
        self.PR = None
        self.NU = None
        # linked list pointers
        self.prev = None
        self.next = None
        OPCODES = {
        "load": 0, "loadI": 1, "store": 2,
        "add": 3, "sub": 4, "mult": 5,
        "lshift": 6, "rshift": 7, "output": 8, "nop": 9
        }
        self.opname = OPCODES[opcode]

    def latency(self):
        if self.opcode in ("load", "store"):
            return 6
        if self.opcode == "mult":
            return 3
        return 1
    
    def units(self):
        if self.opcode in ("load", "store"):
            return {"f0"}
        if self.opcode == "mult":
            return {"f1"}
        return {"f0", "f1"}

    def __repr__(self):
        return f"ILOp(line={self.line}, {self.opcode}, {self.op1},{self.op2},{self.op3})"

#Parser
class Parser:    
    def __init__(self, tokens=None, scanner=None):
        self.errors = 0
        if scanner:
            self.tokens = scanner.scan()
        elif tokens is not None:
            self.tokens = tokens
        else:
            raise ValueError("Parser requires either tokens or a scanner.")

        self.i = 0  # index into token list
        self.ir_list = []
        self.error_found = False
    
    def _error(self, line, message):
        #Prints errors to stderr
        print(f"ERROR {line}: {message}", file=sys.stderr)
        self.errors += 1
    
    def parse(self):
        while self.i < len(self.tokens):
            line_tokens = self._collect_line_tokens()
            if not line_tokens:  # empty line or comment only
                continue
            success, op = self._parse_line(line_tokens)
            if success:
                self.ir_list.append(op)
            else:
                self.error_found = True
        return (not self.error_found), self.ir_list
    
    def _collect_line_tokens(self):
        # Gather tokens until EOL or EOF from self.tokens
        line_tokens = []
        while self.i < len(self.tokens):
            tok = self.tokens[self.i]
            self.i += 1
            if tok[0] == "COMMENT":
                break
            if tok[0] == "EOL":
                break
            line_tokens.append(tok)
        return line_tokens

    def _tok(self, line_tokens, pos):
        """Return token at pos in line_tokens or None if out-of-range."""
        if pos < 0 or pos >= len(line_tokens):
            return None
        return line_tokens[pos]

    def _is_end_or_comment(self, tok):
        return tok is None or tok[0] == "COMMENT"
        
    def _require_kind(self, line_tokens, pos, kinds, line_num, expect_desc):
        tok = self._tok(line_tokens, pos)

        # 1. Missing token
        if tok is None:
            if line_tokens[0][1] in ("load", "store"):
                if kinds == "REGISTER":
                    self._error(line_num, f"Missing {expect_desc} in load or store.")
                elif kinds == "ARROW":
                    self._error(line_num, f"Missing '=>' in load or store.")
                else:
                    self._error(line_num, f"Missing {expect_desc}.")
            else:
                self._error(line_num, f"Missing {expect_desc}.")
            return None

        # Malformed token
        if tok[0] == "ERROR_TOKEN":
            if kinds == "REGISTER" or kinds == "CONST":
                self._error(line_num, f'"{tok[1]}" is not a valid word.')
            elif kinds == "ARROW":
                self._error(line_num, f'"{tok[1]}" is not a valid arrow.')
            elif kinds == "COMMA":
                self._error(line_num, f'"{tok[1]}" is not a valid comma.')
            else:
                self._error(line_num, f'"{tok[1]}" is not valid.')
            if kinds in ("REGISTER", "CONST", "ARROW", "COMMA"):
                if line_tokens[0][1] in ("load","store","loadI"):  # adjust per opcode
                    self._error(line_num, f"Missing {expect_desc} in {line_tokens[0][1]}.")
                else:
                    self._error(line_num, f"Missing {expect_desc}.")
            return None

        if tok[0] not in (kinds if isinstance(kinds, tuple) else (kinds,)):
            # Map error messages to reference
            if kinds == "OPCODE":
                self._error(line_num, f'"{tok[1]}" is not a valid word.')
            elif kinds == "REGISTER":
                if line_tokens[0][1] in ("load","store"):
                    self._error(line_num, f"Missing {expect_desc} in load or store.")
            elif kinds == "CONST":
                self._error(line_num, f"Missing {expect_desc}.")
            elif kinds == "ARROW":
                self._error(line_num, f"Missing '=>' in load or store.")
            elif kinds == "COMMA":
                self._error(line_num, f"Missing comma in {line_tokens[0][1]}.")
            else:
                self._error(line_num, f"Unexpected token '{tok[1]}'")
            return None
        return tok

    def _parse_reg_number(self, tok, line_num):
        #Checks that token is REGISTER and return the numeric register id
        if tok is None:
            return None
        if tok[0] != "REGISTER":
            return None
        lex = tok[1]
        # must be 'r' followed by at least one digit
        if len(lex) < 2 or not lex[1:].isdigit():
            self._error(line_num, f"invalid register '{lex}'")
            return None
        return int(lex[1:])

    def _parse_const_value(self, tok, line_num):
        #Checks token is CONST and in allowed range
        if tok[0] != "CONST":
            self._error(line_num, f"unexpected token '{tok[1]}' — expected a constant")
            return None
        try:
            val = int(tok[1])
        except Exception:
            self._error(line_num, f"invalid constant '{tok[1]}'")
            return None
        if val < 0 or val > 2**31 - 1:
            self._error(line_num, f"constant out of range '{tok[1]}'")
            return None
        return val
    
    def _parse_line(self, line_tokens):
        #Parse a single line (list of tokens up to but not including EOL).
        if not line_tokens:
            return False, None

        line_num = line_tokens[0][2]
        first = line_tokens[0]

        # must start with OPCODE
        if first[0] != "OPCODE":
            self._error(line_num, f"unexpected token '{first[1]}' — expected opcode")
            return False, None

        opcode = first[1]

        # enforce whitespace after opcode
        """
        if len(line_tokens) >= 2:
            next_tok = line_tokens[1]
            if next_tok.col <= first.col + len(first[1]):
                self._error(line_num, f"missing whitespace after opcode '{opcode}' — opcode must be followed by blank(s)")
                return False, None
        """

        # dispatch to opcode-specific parser
        if opcode in {"add", "sub", "mult", "lshift", "rshift"}:
            return self._parse_add_like(opcode, line_tokens)
        elif opcode == "load":
            return self._parse_load(line_tokens)
        elif opcode == "loadI":
            return self._parse_loadI(line_tokens)
        elif opcode == "store":
            return self._parse_store(line_tokens)
        elif opcode == "output":
            return self._parse_output(line_tokens)
        elif opcode == "nop":
            return self._parse_nop(line_tokens)
        else:
            self._error(line_num, f"unknown opcode '{opcode}'")
            return False, None
        

    #Parse_line helper functions
    def _parse_add_like(self, opcode, tokens):
        line_num = tokens[0][2]
        # Expect: OPCODE REG COMMA REG ARROW REG
        if len(tokens) < 6:
            self._error(line_num, "Missing token(s) in add instruction.")
            return False, None

        # Validate token types in order
        t1, t2, t3, t4, t5 = tokens[1:6]

        if t1[0] != "REGISTER":
            self._error(line_num, "First operand must be a register")
            return False, None
        if t2[0] != "COMMA":
            self._error(line_num, "Missing comma in add.")
            return False, None
        if t3[0] != "REGISTER":
            self._error(line_num, "Second operand must be a register")
            return False, None
        if t4[0] != "ARROW":
            self._error(line_num, "Missing '=>' in add.")
            return False, None
        if t5[0] != "REGISTER":
            self._error(line_num, "Destination must be a register")
            return False, None

        # Skip any extra tokens beyond comment
        if len(tokens) > 6 and tokens[6][0] != "COMMENT":
            self._error(line_num, "Too many tokens on line")
            return False, None

        # Construct ILOperation
        src1 = int(t1[1][1:])
        src2 = int(t3[1][1:])
        dst = int(t5[1][1:])
        return True, ILOperation(line_num, opcode, src1, src2, dst)
         
    def _parse_load(self, tokens):
        line_num = tokens[0][2]
        t1 = self._require_kind(tokens, 1, "REGISTER", line_num, "a register (source)")
        if t1 is None:
            return False, None
        t2 = self._require_kind(tokens, 2, "ARROW", line_num, "'=>'")
        if t2 is None:
            return False,None
        t3 = self._require_kind(tokens, 3, "REGISTER", line_num, "a register (destination)")
        if t3 is None:
            return False, None
        src = self._parse_reg_number(t1, line_num) if t1 else None
        dst = self._parse_reg_number(t3, line_num) if t3 else None
        if len(tokens) > 4 and tokens[4][0] != "COMMENT":
            self._error(line_num, f"unexpected token '{tokens[4][1]}' — too many tokens on line")
            return False, None
        if None in (src, dst):
            return False, None
        return True, ILOperation(line_num, "load", src, None, dst)


    def _parse_loadI(self, tokens):
        line_num = tokens[0][2]

        # Constant
        t1 = self._require_kind(tokens, 1, "CONST", line_num, "a constant")
        if t1 is None:
            return False, None

        # Arrow
        t2 = self._require_kind(tokens, 2, "ARROW", line_num, "'=>'")
        if t2 is None:
            return False, None

        t3 = self._require_kind(tokens, 3, "REGISTER", line_num, "a register")
        if t3 is None:
            return False, None

        val = self._parse_const_value(t1, line_num) if t1 and t1[0] != "ERROR_TOKEN" else None
        dst = self._parse_reg_number(t3, line_num) if t3 and t3[0] != "ERROR_TOKEN" else None


        if 4 < len(tokens) and tokens[4][0] != "COMMENT":
            self._error(line_num, f"unexpected token '{tokens[4][1]}' — too many tokens on line")
            return False, None

        if None in (val, dst):
            return False, None

        return True, ILOperation(line_num, "loadI", val, None, dst)


    def _parse_store(self, tokens):
        line_num = tokens[0][2]
        t1 = self._require_kind(tokens, 1, "REGISTER", line_num, "a register (source)")
        if t1 is None:
            return False, None
        t2 = self._require_kind(tokens, 2, "ARROW", line_num, "'=>'")
        if t2 is None:
            return False, None
        t3 = self._require_kind(tokens, 3, "REGISTER", line_num, "a register (destination)")
        if t3 is None:
            return False, None
        if len(tokens) > 4 and tokens[4][0] != "COMMENT":
            self._error(line_num, f"unexpected token '{tokens[4][1]}' — too many tokens on line")
            return False, None
        src = self._parse_reg_number(t1, line_num) if t1 else None
        dst = self._parse_reg_number(t3, line_num) if t3 else None
        if None in (src, dst):
            return False, None
        return True, ILOperation(line_num, "store", src, None, dst)


    def _parse_output(self, tokens):
        line_num = tokens[0][2]
        t1 = self._require_kind(tokens, 1, "CONST", line_num, "a constant")
        if not t1:
            return False, None
        if len(tokens) > 2 and tokens[2][0] != "COMMENT":
            self._error(line_num, f"unexpected token '{tokens[2][1]}' — too many tokens on line")
            return False, None
        val = self._parse_const_value(t1, line_num)
        if val is None:
            return False, None
        return True, ILOperation(line_num, "output", val, None, None)


    def _parse_nop(self, tokens):
        line_num = tokens[0][2]
        if len(tokens) > 1 and tokens[1][0] != "COMMENT":
            self._error(line_num, f"unexpected token '{tokens[1][1]}' — 'nop' takes no operands")
            return False, None
        return True, ILOperation(line_num, "nop", None, None, None)
    
    
def help_info():
    # print("Usage: ./412fe [mode] <filename>")
    # print("Modes:")
    # print(" k   : 3 < k < 64. Reallocates registers with k physical registers")
    # print (" -x : Rename input block and print to stdout. For code check 1.")
    # print (" -h : View help")

    print("Usage: ./schedule <filename>")
    print("Shows operation schedule for input ILOC file")
    print (" -h : View help")


    #print("  -s <file>   : Scan the file and print tokens")
    #print("  -p <file>   : Parse the file and report success/errors (default)")
    #print("  -r <file>   : Parse and print the intermediate representation")
    sys.exit(0)

def read_operands(op):
    opn = op.opname
    if 3 <= opn <= 7:
        # src1, src2, dest
        return [op.op1, op.op2]
    elif opn == 0:
        return [op.op1]
    elif opn == 2:
        return [op.op1, op.op3]
    else:
        return []

def write_operands(op):
    opn = op.opname
    if 3 <= opn <= 7 or opn == 0 or opn == 1:
        # src1, src2, dest
        return [op.op3]
    else:
        return []

def build_dependence_graph(ir_list):
    n = len(ir_list)
    preds = [set() for _ in range(n)]
    succs = [set() for _ in range(n)]

    last_def = {}     # reg -> index of last defining op
    last_use = {}     # reg -> index of last using op
    last_store = None
    last_output = None
    store_order = []
 
    for i, op in enumerate(ir_list):
        # True dependencies (RAW)
        for r in read_operands(op):
            #opn = op.opn
            if r in last_def:
                preds[i].add(last_def[r])
                succs[last_def[r]].add(i)
            #if opn == 0 or opn == 2: #Load or store
                
        # Output dependencies (WAW)
        for r in write_operands(op):
            if r in last_def and last_def[r] != i:   # avoid self-dependence for repeated writes
                preds[i].add(last_def[r])
                succs[last_def[r]].add(i)
        # Anti-dependencies (WAR)
        # for r in write_operands(op):
        #     if r in last_use and last_use[r] != i:
        #         preds[i].add(last_use[r])
        #         succs[last_use[r]].add(i)
        if op.opcode == "store":
            # if last_store is not None:
            #     preds[i].add(last_store)
            #     succs[last_store].add(i)
            store_order.append(i)

            if last_output is not None:
                preds[i].add(last_output)
                succs[last_output].add(i)

            last_store = i

        elif op.opcode == "load":
            # load must wait for:
            #  - previous store (true dependency)
            if last_store is not None:
                preds[i].add(last_store)
                succs[last_store].add(i)

        elif op.opcode == "output":
            # Output must wait for all prior memory ops
            if last_store is not None:
                preds[i].add(last_store)
                succs[last_store].add(i)
            if last_output is not None:
                preds[i].add(last_output)
                succs[last_output].add(i)
            last_output = i

        # Update last_use/last_def
        for r in write_operands(op):
            last_def[r] = i
        for r in read_operands(op):
            last_use[r] = i

    return preds, succs, store_order

def list_schedule(ir_list):
    n = len(ir_list)
    if n == 0:
        return []

    preds, succs, store_order = build_dependence_graph(ir_list)


    in_deg = [len(preds[i]) for i in range(n)]
    ready = {i for i in range(n) if in_deg[i] == 0}

    # earliest issue time from latency constraints
    earliest = [0] * n

    scheduled = [None] * n  # map op index -> (cycle, slot0_or_1)
    cycle_ops = []          # list of lists: each entry is [op_index] or [op_index1, op_index2]

    cur_cycle = 0
    unscheduled = set(range(n))

    # convenience predicates
    def can_issue_in_cycle(i, c):
        return earliest[i] <= c

    while unscheduled:
        # collect candidates that can issue at this cycle
        candidates = [i for i in ready if can_issue_in_cycle(i, cur_cycle)]

        # resource usage flags for this cycle
        have_f0 = False
        have_f1 = False
        have_output = False

        chosen = []

        # priority: longer latency first, then more successors, then original line
        candidates.sort(
            key=lambda i: (-ir_list[i].latency(), -len(succs[i]), ir_list[i].line)
        )

        has_store = False
        for i in candidates:
            if len(chosen) == 2:
                break
            op = ir_list[i]

            opc = op.opcode

            # special constraint: only one output per cycle
            if opc == "output":
                if have_output:
                    continue

            if opc == "store":
                if has_store or store_order[0] != i:
                    continue

            u = op.units()
            # try to assign f0 or f1 without conflict
            if "f1" in u and not have_f1:
                have_f1 = True
                if op.opcode == "output":
                    have_output = True
                chosen.append(i)
            elif "f0" in u and not have_f0:
                have_f0 = True
                if opc == "store":
                    has_store = True
                    store_order.pop(0)
                if op.opcode == "output":
                    have_output = True
                chosen.append(i)
            else:
                # cannot schedule due to unit conflict
                continue

        if not chosen:
            # no ready op can issue in this cycle; insert a nop
            cycle_ops.append(None)  # mark nop cycle
            cur_cycle += 1
            continue

        # record chosen ops in this cycle
        cycle_ops.append(chosen)
        for idx_in_cycle, i in enumerate(chosen):
            scheduled[i] = (cur_cycle, idx_in_cycle)
            unscheduled.remove(i)
            ready.remove(i)

        # update successors: in-degree and earliest times
        for i in chosen:
            op_i = ir_list[i]
            finish_time = cur_cycle + op_i.latency()
            for s in succs[i]:
                in_deg[s] -= 1
                earliest[s] = max(earliest[s], finish_time)
                if in_deg[s] == 0:
                    ready.add(s)

        cur_cycle += 1

    # build scheduled list of ILOOperation *cycles*, then flatten for printing
    scheduled_cycles = []

    for entry in cycle_ops:
        if entry is None:
            # nop cycle
            scheduled_cycles.append([ILOperation(0, "nop", None, None, None)])
        else:
            scheduled_cycles.append([ir_list[i] for i in entry])

    return scheduled_cycles

def format_op(op):
    opc = op.opcode
    if opc in ("add", "sub", "mult", "lshift", "rshift"):
        return f"{opc} r{op.op1}, r{op.op2} => r{op.op3}"
    elif opc == "load":
        return f"{opc} r{op.op1} => r{op.op3}"
    elif opc == "store":
        return f"{opc} r{op.op1} => r{op.op3}"
    elif opc == "loadI":
        return f"{opc} {op.op1} => r{op.op3}"
    elif opc == "output":
        return f"{opc} {op.op1}"
    elif opc == "nop":
        return "nop"
    else:
        return f"# Unknown opcode {opc}"


def print_schedule(scheduled_cycles):
    for cycle in scheduled_cycles:
        if len(cycle) == 1:
            print(format_op(cycle[0]))
        elif len(cycle) == 2:
            left = format_op(cycle[0])
            right = format_op(cycle[1])
            print(f"[ {left} ; {right} ]")
        else:
            # should not happen, but fall back to sequential
            for op in cycle:
                print(format_op(op))

def rename_ir_ssa(ir_list):
    current = {}   # old reg -> current new reg
    next_reg = 0
    renamed = []

    def get_current(old):
        # use current mapping; assume all needed defs appear before uses
        return current[old]

    def new_name_for(old):
        nonlocal next_reg
        nr = next_reg
        next_reg += 1
        current[old] = nr
        return nr

    for op in ir_list:
        opc = op.opcode
        if opc in ("add", "sub", "mult", "lshift", "rshift"):
            # uses: op1, op2; def: op3
            src1 = get_current(op.op1)
            src2 = get_current(op.op2)
            dst  = new_name_for(op.op3)
            renamed.append(ILOperation(op.line, opc, src1, src2, dst))

        elif opc == "load":
            # use: base (op1); def: op3
            base = get_current(op.op1)
            dst  = new_name_for(op.op3)
            renamed.append(ILOperation(op.line, opc, base, None, dst))

        elif opc == "store":
            # uses: value (op1), addr (op3); no def
            val  = get_current(op.op1)
            addr = get_current(op.op3)
            renamed.append(ILOperation(op.line, opc, val, None, addr))

        elif opc == "loadI":
            # const -> def
            dst = new_name_for(op.op3)
            renamed.append(ILOperation(op.line, opc, op.op1, None, dst))

        elif opc == "output":
            # const only
            renamed.append(ILOperation(op.line, opc, op.op1, None, None))

        elif opc == "nop":
            renamed.append(ILOperation(op.line, opc, None, None, None))

        else:
            # should not happen in this lab
            renamed.append(op)

    return renamed

if __name__ == "__main__":
    #start = time.time()
    args = sys.argv[1:]
    if not args or "-h" in args:
        help_info()  # prints help and exits

    if len(args) != 1:
        print("Usage: schedule <input_file>", file=sys.stderr)
        sys.exit(1)

    filename = args[0]
    try:
        scanner = Scanner(filename)
        parser = Parser(scanner=scanner)
        success, ir_list = parser.parse()
    except FileNotFoundError:
        print(f"ERROR: file '{filename}' not found", file=sys.stderr)
        sys.exit(1)

    if not success:
        sys.exit(1)

    ir_list = [op for op in ir_list if op.opcode != "nop"]
    #allocated_ir = allocate_registers(ir_list, k)
    #print_allocated(allocated_ir)

    ir_list_renamed = rename_ir_ssa(ir_list)

    scheduled_cycles = list_schedule(ir_list_renamed)
    print_schedule(scheduled_cycles)


    #end = time.time()
    #print(f"Elapsed time: {end - start:.4f} seconds")