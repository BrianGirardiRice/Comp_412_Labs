import sys
#import time

"""
#Token Class
class Token:
    def __init__(self, category, lexeme, line):
        self.category = category
        self.lexeme = lexeme
        self.line = line

    def __str__(self):
        return f"{self.line}: {self.category} -> {self.lexeme}"
"""
    

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
    print("Usage: ./412fe [mode] <filename>")
    print("Modes:")
    print("  -h          : Show this help message")
    print("  -s <file>   : Scan the file and print tokens")
    print("  -p <file>   : Parse the file and report success/errors (default)")
    print("  -r <file>   : Parse and print the intermediate representation")
    sys.exit(0)


#Performs renaming and prints the renamed ILOC block to stdout.
def rename_ir_map(ir_list):
    mapping = {}   # old_reg_number (int) -> new index (int)
    next_r = 0

    """Local function to ensure old (int) is in mapping; return its new index (int)."""
    def ensure(old):
        nonlocal next_r
        if old is None:
            return None
        if old not in mapping:
            mapping[old] = next_r
            next_r += 1
        return mapping[old]

    for op in ir_list:
        for r in get_register_operands(op):
            ensure(r)
        """
        opc = op.opcode
        if opc in ("add", "sub", "mult", "lshift", "rshift"):
            # src1, src2, dest
            ensure(op.op1); ensure(op.op2); ensure(op.op3)
        elif opc == "load":
            # source, dest
            ensure(op.op1); ensure(op.op3)
        elif opc == "store":
        # source, dest
            ensure(op.op1); ensure(op.op3)
        elif opc == "loadI":
            # op.op1 is a constant
            ensure(op.op3)
    """
    
    return mapping

def print_renamed(ir_list, mapping):
    # Print renamed ILOC lines
    for op in ir_list:
        opc = op.opcode
        if opc in ("add", "sub", "mult", "lshift", "rshift"):
            a = f"r{mapping[op.op1]}"
            b = f"r{mapping[op.op2]}"
            c = f"r{mapping[op.op3]}"
            print(f"{opc} {a}, {b} => {c}")
        elif opc == "load":
            a = f"r{mapping[op.op1]}"
            c = f"r{mapping[op.op3]}"
            print(f"{opc} {a} => {c}")
        elif opc == "store":
            a = f"r{mapping[op.op1]}"
            c = f"r{mapping[op.op3]}"
            print(f"{opc} {a} => {c}")
        elif opc == "loadI":
            # op.op1 is an integer constant
            c = f"r{mapping[op.op3]}"
            print(f"{opc} {op.op1} => {c}")
        elif opc == "output":
            # op.op1 is an integer constant for output
            print(f"{opc} {op.op1}")
        elif opc == "nop":
            print("nop")
        else:
            print(f"# Unknown opcode {opc}", file=sys.stderr)

"""
def allocate_and_emit_ir(ir_list, num_phys = 16):
    spill_regs = [num_phys - 2, num_phys - 1] 
    allocatable = num_phys-2 

    active = [] # list of touples (virt_reg, end, phys)
    reg_map = {} # maps virt_reg to ("phys"/"spill", num/addr)
    emitted_ir = []

    live_ranges = compute_live_ranges(ir_list)
    for idx, op in enumerate(ir_list):
        reg_operands == get_register_operands(op)
        active = [(vr, e, phys) for (vr, e, phys) in active if e >= start]
        #find free physicals
        used = {phys for (_ ,_ , phys) in active}
        free_regs = [r for r in range(allocatable) if r not in used]
        
        # Helper: get physical register or insert spill code
        def phys_or_spill(vr, is_read):
            if vr is None:
                return None
            kind, val = reg_map.get(vr, ("phys", vr))
            if kind == "phys":
                return f"r{val}"
            else:
                # need to load/store from spill
                addr = val
                if is_read:
                    emitted_ir.append(ILOperation(op.line, "loadI", addr, None, spill_regs[0]))
                    emitted_ir.append(ILOperation(op.line, "load", spill_regs[0], None, spill_regs[1]))
                    return f"r{spill_regs[1]}"
                else:
                    emitted_ir.append(ILOperation(op.line, "store", spill_regs[1], None, spill_regs[0]))
                    emitted_ir.append(ILOperation(op.line, "loadI", addr, None, spill_regs[1]))
                    return f"r{spill_regs[1]}"

        # Assign registers for operands/dest
        for vr in [op.op1, op.op2, op.op3]:
            if vr is None or vr in reg_map:
                continue
            if free_regs:
                phys = free_regs.pop(0)
                reg_map[vr] = ("phys", phys)
                # Add to active
                start, end = live_ranges[vr]
                active.append((vr, end, phys))
            else:
                # Spill victim with furthest end
                victim = max(active, key=lambda x: x[1])
                spill_addr = next_spill_addr
                next_spill_addr += 4
                if victim[1] > live_ranges[vr][1]:
                    # spill victim
                    reg_map[victim[0]] = ("spill", spill_addr)
                    active.remove(victim)
                    phys = victim[2]
                    reg_map[vr] = ("phys", phys)
                    active.append((vr, live_ranges[vr][1], phys))
                else:
                    # spill current vr
                    reg_map[vr] = ("spill", spill_addr)

        # Emit instruction using current mapping
        opc = op.opcode
        a = phys_or_spill(op.op1, True) if hasattr(op, 'op1') else None
        b = phys_or_spill(op.op2, True) if hasattr(op, 'op2') else None
        c = phys_or_spill(op.op3, False) if hasattr(op, 'op3') else None

        emitted_ir.append(ILOperation(op.line, opc, a, b, c))

    return emitted_ir
"""


# Abstracts code for rename_ir_map and allocate_registers.
# returns a list of operands in op that are registers
def get_register_operands(op):
    opc = op.opcode
    if opc in ("add", "sub", "mult", "lshift", "rshift"):
        # src1, src2, dest
        return [op.op1, op.op2, op.op3]
    elif opc == "load":
        # source, dest
        return [op.op1, op.op3]
    elif opc == "store":
    # source, dest
        return [op.op1, op.op3]
    elif opc == "loadI":
        # op.op1 is a constant
        return [op.op3]
    else:
        return []

def read_operands(op):
    opc = op.opcode
    if opc in ("add", "sub", "mult", "lshift", "rshift"):
        # src1, src2, dest
        return [op.op1, op.op2]
    elif opc in ("load"):
        return [op.op1]
    elif opc == "store":
        return [op.op1, op.op3]
    else:
        return []

def write_operands(op):
    opc = op.opcode
    if opc in ("add", "sub", "mult", "lshift", "rshift"):
        # src1, src2, dest
        return [op.op3]
    elif opc in ["load", "loadI"]:
        # source, dest
        return [op.op3]
    else:
        return []


# Helper for allocate_register, gets start_end ranges for registers
def compute_live_ranges(ir_list):
    intervals = []
    live_ranges = {}
    for idx, op in enumerate(ir_list):
        for r in read_operands(op):
            if r in live_ranges:
                opc = op.opcode
                if opc == "store":
                    live_ranges[r][1] = 2*idx+1
                else:
                    live_ranges[r][1] = 2*idx
            elif opc == "store":
                live_ranges[r] = [2*idx+1, 2*idx+1]
            else:
                live_ranges[r] = [2*idx, 2*idx]
        for r in write_operands(op):
            if r in live_ranges:
                intervals.append((r, live_ranges[r][0], live_ranges[r][1]))
            live_ranges[r] = [2*idx+1, 2*idx+1]
    for r in live_ranges:
        intervals.append((r, live_ranges[r][0], live_ranges[r][1]))
    return intervals

def linear_scan_and_emit(intervals, num_phys):
    intervals_with_flag = [(reg, start, end, 1) for (reg, start, end) in intervals]
    sorted_intervals = sorted(intervals_with_flag, key=lambda x: x[1])
    detachable_intervals = sorted_intervals.copy()

    # reserves two spots for spills
    spill_load = num_phys - 1
    spill_store = num_phys - 1
    allocatable = num_phys-1

    active = [] # list of touples (virt_reg, end, phys)
    reg_map = {} # maps virt_reg to ("phys"/"spill", num/addr)
    next_spill_addr = 32768
    allocated_ir = []

    spill_slot_map = {}
    def get_spill_slot(vr):
        nonlocal next_spill_addr
        if vr not in spill_slot_map:
            spill_slot_map[vr] = next_spill_addr
            next_spill_addr += 4
        return spill_slot_map[vr]
        
    def add_reg_to_map(vr, end):
        nonlocal next_spill_addr
        #find free physicals
        candidates = [v for v in active if v[3] == 1]  
        used = {phys for (_ ,_ , phys, _) in candidates}
        free_regs = [r for r in range(allocatable) if r not in used and r not in busy]
        if free_regs:
            #assign first free
            phys = free_regs[0]
            reg_map[vr] = ("phys", phys)
            return phys
        else:
            #spill needed
            spill_possibilities = [(n, e, pr, flag) for (n, e, pr, flag) in candidates if pr not in busy]
            victim = max(spill_possibilities, key=lambda x: x[1]) #furthest away end
            spill_addr = get_spill_slot(victim[0])
            # victim ends after current interval, spill
            prefix.append(ILOperation(op.line, "loadI", spill_addr, None, f"r{spill_store}"))
            prefix.append(ILOperation(op.line, "store", f"r{victim[2]}", None, f"r{spill_store}"))
            reg_map[victim[0]] = ("spill", spill_addr)
            #replace victim in active
            active.remove(victim)
            victim = (victim[0], victim[1], victim[2], -1)
            active.append(victim)
            phys = victim[2]
            reg_map[vr] = ("phys", phys)
            return phys
        

    # load_check = True when reading the register, false when writing to it
    def phys_or_load_or_store(r, load_check=True):
        if r is None:
            return None
        kind, val = reg_map.get(r, ("phys", r))
        if kind == "phys":
            return f"r{val}"
        else:
            #retrieval
            addr = spill_slot_map[r]
            phys = add_reg_to_map(r, end=idx)
            if phys is None:
                # VR itself is spilled; we can’t restore to a phys register yet
                return f"spill{reg_map[r][1]}"
            # otherwise, restore from memory
            for interval in active:
                if interval[0] == r and interval[3] == -1:
                    new_interval = (interval[0], interval[1], interval[2], 1)
                    active.remove(interval)
                    active.append(new_interval)

            prefix.append(ILOperation(op.line, "loadI", addr, None, f"r{spill_load}"))
            prefix.append(ILOperation(op.line, "load", f"r{spill_load}", None, f"r{phys}"))
            busy.append(phys)

            reg_map[r] = ("phys", phys)
            return f"r{phys}"

    def expire_old(current_start):
        #Expire intervals that end before the given start position.
        nonlocal active, reg_map
        new_active = []
        for (vr, end, phys, spilled) in active:
            if end >= current_start:
                new_active.append((vr, end, phys, spilled))
            else:
                # this virtual register's live range ended before now
                # free its mapping
                if reg_map.get(vr, (None, None))[0] == "phys":
                    del reg_map[vr]
        active = new_active


    def prep_write():
        expire_old(2*idx+1)
        expand_active(2*idx+1)
    
    def prep_read():
        expire_old(2*idx)
        expand_active(2*idx)


    def expand_active(threshold):
        while detachable_intervals and detachable_intervals[0][1] == threshold:
            my_interval = detachable_intervals.pop(0)
            phys = add_reg_to_map(my_interval[0], my_interval[2])
            if phys is not None:
                active.append((my_interval[0], my_interval[2], phys, 1))

    expand_active(0)
    for idx, op in enumerate(ir_list):
        opc = op.opcode
        prefix = []
        suffix = []
        busy = []
        for busy_op in (op.op1, op.op2):
            if busy_op in reg_map and reg_map[busy_op][0] == "phys":
                busy.append(reg_map[busy_op][1])
        prep_read()
        if opc in ("add", "sub", "mult", "lshift", "rshift"):
            a = phys_or_load_or_store(op.op1, True)
            b = phys_or_load_or_store(op.op2, True)
            prep_write()
            c = phys_or_load_or_store(op.op3, False)
            allocated_ir.extend(prefix)
            allocated_ir.append(ILOperation(op.line, opc, a, b, c))
            allocated_ir.extend(suffix)
        elif opc == "load":
            a = phys_or_load_or_store(op.op1, True)
            prep_write()
            c = phys_or_load_or_store(op.op3, False)
            allocated_ir.extend(prefix)
            allocated_ir.append(ILOperation(op.line, opc, a, None, c))
            allocated_ir.extend(suffix)
        elif opc == "store":
            a = phys_or_load_or_store(op.op1, True)
            prep_write()
            c = phys_or_load_or_store(op.op3, True)
            allocated_ir.extend(prefix)
            allocated_ir.append(ILOperation(op.line, opc, a, None, c))
            allocated_ir.extend(suffix)
        elif opc == "loadI":
            prep_write()
            c = phys_or_load_or_store(op.op3, False)
            allocated_ir.extend(prefix)
            allocated_ir.append(ILOperation(op.line, opc, op.op1, None, c))
            allocated_ir.extend(suffix)
        else:
            allocated_ir.append(op)
         


    return allocated_ir    

# apply a renaming pass to IR so virtual registers are dense 0..N-1
def apply_rename_mapping(ir_list, mapping):
    # produce a new IR list (or mutate) using mapping for register operands
    new_ir = []
    for op in ir_list:
        opc = op.opcode
        if opc in ("add", "sub", "mult", "lshift", "rshift"):
            new_op1 = mapping.get(op.op1, None)
            new_op2 = mapping.get(op.op2, None)
            new_op3 = mapping.get(op.op3, None)
            new_ir.append(ILOperation(op.line, opc, new_op1, new_op2, new_op3))
        elif opc == "load":
            new_op1 = mapping.get(op.op1, None)
            new_op3 = mapping.get(op.op3, None)
            new_ir.append(ILOperation(op.line, opc, new_op1, None, new_op3))
        elif opc == "store":
            new_op1 = mapping.get(op.op1, None)
            new_op3 = mapping.get(op.op3, None)
            new_ir.append(ILOperation(op.line, opc, new_op1, None, new_op3))
        elif opc == "loadI":
            # op.op1 is constant, op.op3 is destination reg
            new_op3 = mapping.get(op.op3, None)
            new_ir.append(ILOperation(op.line, opc, op.op1, None, new_op3))
        else:
            # output, nop or others
            new_ir.append(ILOperation(op.line, opc, op.op1, op.op2, op.op3))
    return new_ir


#Linear resister allocation, maps renamed registers (r0, r1, ...) to physical registers
def allocate_registers(ir_list, num_phys=16):    
    intervals = compute_live_ranges(ir_list)
    return linear_scan_and_emit(intervals, num_phys)


def print_allocated(ir_list):
    for op in ir_list:
        opc = op.opcode
        a = op.op1
        b = op.op2
        c = op.op3

        if opc in ("add", "sub", "mult", "lshift", "rshift"):
            print(f"{opc} {a}, {b} => {c}")
        elif opc == "load":
            print(f"{opc} {a} => {c}")
        elif opc == "store":
            print(f"{opc} {a} => {c}")
        elif opc == "loadI":
            # op1 is constant
            print(f"{opc} {a} => {c}")
        elif opc == "output":
            print(f"{opc} {a}")
        elif opc == "nop":
            print("nop")
        else:
            print(f"# Unknown opcode {opc}", file=sys.stderr)

if __name__ == "__main__":
    #start = time.time()
    args = sys.argv[1:]
    if "-h" in args:
        help_info()  # prints help and exits

    if len(args) != 2:
        print("Usage: 412alloc k <input_file>", file=sys.stderr)
        sys.exit(1)

    try:
        k = int(sys.argv[1])
    except ValueError:
        print(f"Invalid register count '{sys.argv[1]}'", file=sys.stderr)
        sys.exit(1)

    if k < 3 or k > 64:
        print(f"Register count k must be between 3 and 64, got {k}", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[2]

    scanner = Scanner(filename)
    parser = Parser(scanner=scanner)
    success, ir_list = parser.parse()
    if not success:
        sys.exit(1)
    allocated_ir = allocate_registers(ir_list, k)
    print_allocated(allocated_ir)


    #end = time.time()
    #print(f"Elapsed time: {end - start:.4f} seconds")