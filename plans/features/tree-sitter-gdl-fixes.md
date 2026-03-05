# tree-sitter-gdl: Parse Error Report

Systematic testing of `tree-sitter-gdl` against IDL/GDL constructs found in TCV analysis code. Grammar source: `~/Code/tree-sitter-gdl/grammar.js` (503 lines).

## Summary

**81% of common IDL patterns parse without errors** (21/26 test cases). Five categories of construct produce parse errors, all traceable to missing grammar rules.

## Error Categories

### 1. Struct Member Assignment (High Impact)

Reading `s.field` works via `member_expression`, but **assigning to `s.field`** fails because `assignment.left` only allows `identifier | subscript_expression` — not `member_expression`.

```idl
; FAILS — s.value is a member_expression, not valid as assignment target
pro t
  s.value = 100
end

; WORKS — reading struct members is fine
pro t, s
  x = s.value
end
```

**Fix:** Add `member_expression` to the `assignment.left` field choices in `grammar.js`:

```js
assignment: $ => prec.right(seq(
  field('left', choice($.identifier, $.subscript_expression, $.member_expression)),
  ...
)),
```

**Impact:** Very common in TCV code. Every struct-heavy procedure uses `self.field = value` or `params.shot = shot`.

### 2. Pointer Dereference `*ptr` (High Impact)

The `*` token is defined as the multiplication operator in `binary_expression` but there is no unary `*` (dereference) in `unary_expression`. The parser sees `*ptr` and tries to parse it as multiplication with a missing left operand.

```idl
; ALL FAIL
x = *p          ; dereference
(*p)[0] = 99    ; dereference + subscript
*p = 42         ; dereference + assign
```

**Fix:** Add pointer dereference to `unary_expression` and `assignment.left`:

```js
unary_expression: $ => choice(
  prec(PREC.UNARY, seq('-', $._expression)),
  prec(PREC.UNARY, seq('+', $._expression)),
  prec(PREC.NOT, seq(kw('not'), $._expression)),
  prec(PREC.UNARY, seq('~', $._expression)),
  prec(PREC.UNARY, seq('*', $._expression)),  // pointer dereference
),

assignment: $ => prec.right(seq(
  field('left', choice(
    $.identifier,
    $.subscript_expression,
    $.member_expression,
    $.unary_expression,  // for *ptr = value
  )),
  ...
)),
```

This will conflict with `*` as multiplication. Needs a precedence strategy — likely `prec.dynamic` or a conflict declaration since `* expr` as unary can only appear in statement-initial position or after `=`.

**Impact:** Common in TCV code that uses pointer arrays (`ptrarr`, `ptr_new`). The `crpp_tdi_get.pro` pattern `*data[i] = mds$value(...)` is a canonical example.

### 3. System Variable Assignment `!var = expr` (Medium Impact)

`system_variable` (`!p`, `!x`, `!dpi`) is recognized as an expression for reading, but it is not a valid assignment target. The grammar's `assignment.left` doesn't include `system_variable` or `member_expression` on a system variable.

```idl
; FAILS
!p.multi = [0, 2, 1]
!x.range = [0, 100]

; WORKS — reading system variables
x = !dpi
x = !values.f_nan
print, !path
```

**Fix:** Allow system_variable (and member access on system variables) as assignment targets:

```js
assignment: $ => prec.right(seq(
  field('left', choice(
    $.identifier,
    $.subscript_expression,
    $.member_expression,  // covers !p.multi via system_variable.member
    $.system_variable,    // covers !null = expr
  )),
  ...
)),
```

Note: `!p.multi` parses as `member_expression(system_variable("!p"), "multi")` — so adding `member_expression` to assignment targets (fix #1) should cover this case too.

**Impact:** Medium. System variable configuration (`!p.multi`, `!x.range`) appears in plotting setup code. Less common than struct assignment but still frequent.

### 4. Arrow Method Calls `obj->Method` (High Impact)

The `method_call` rule requires parentheses: `obj->Method(args)`. IDL allows calling methods as procedures (no parens, comma-separated args) — `obj->Draw` or `obj->SetProperty, color=red`.

```idl
; FAILS — no parentheses
obj->Draw
obj->SetProperty, color=[255,0,0]
self.window->Draw, self.view

; WORKS — with parentheses (function-style)
result = obj->GetProperty(name)
```

**Fix:** Split `method_call` into two forms — function-style (with parens, returns value) and procedure-style (no parens, statement):

```js
// Expression form: obj->Method(args) — returns a value
method_call: $ => prec.left(PREC.METHOD, seq(
  $._expression,
  '->',
  $.identifier,
  '(',
  optional($.argument_list),
  ')',
)),

// Statement form: obj->Method, args — procedure call via method
method_procedure_call: $ => prec.right(seq(
  $._expression,
  '->',
  $.identifier,
  optional(seq(',', $.argument_list)),
)),
```

Add `method_procedure_call` to `_simple_statement`.

**Impact:** Very high for OOP IDL code. TCV diagnostic readers use `self->method` extensively. The `ece_reader::read` example has 4 arrow calls, all failing.

### 5. Double-Colon Method Definitions `Class::Method` (High Impact)

Function/procedure definitions using `::` notation for class methods fail. The grammar expects `identifier` as the function name, but `MyClass::Init` contains `::` which isn't part of the identifier pattern.

```idl
; FAILS
function MyClass::Init, data
  self.data = ptr_new(data)
  return, 1
end

; FAILS  
pro MyClass::Cleanup
  ptr_free, self.data
end
```

**Fix:** Allow qualified names in function/procedure definitions:

```js
qualified_name: $ => seq($.identifier, '::', $.identifier),

procedure_definition: $ => seq(
  kw('pro'),
  field('name', choice($.identifier, $.qualified_name)),
  ...
),

function_definition: $ => seq(
  kw('function'),
  field('name', choice($.identifier, $.qualified_name)),
  ...
),
```

**Impact:** High for OOP IDL. Every class definition file has `__define`, `::Init`, `::Cleanup`, and custom methods.

## Real TCV Code Results

| File Pattern | Errors | Root Cause |
|---|---|---|
| `liuqe_params.pro` (function, mdsvalue, structs) | 0 | — |
| `crpp_tdi_get.pro` (pointer deref, !null assign) | 2 | #2, #3 |
| `ece_reader::read` (class method, self.field, self->) | 5 | #1, #4, #5 |

## Construct Coverage Matrix

| Construct | Status | Category |
|---|---|---|
| `pro/function` definitions | OK | — |
| `for/while/repeat` loops | OK | — |
| `if/then/else` (inline + block) | OK | — |
| `case/switch` | OK | — |
| Keyword arguments (`key=val`, `/flag`) | OK | — |
| String operations | OK | — |
| Array indexing `a[0:5]` | OK | — |
| Struct creation `{field: val}` | OK | — |
| Struct read `s.field` | OK | — |
| `common` blocks | OK | — |
| `compile_opt` | OK | — |
| Line continuation `$` | OK | — |
| `mdsopen/mdsvalue/mds$open` | OK | — |
| TDI calls | OK | — |
| Ternary `? :` | OK | — |
| All comparison/logical operators | OK | — |
| **Struct assign `s.field = val`** | FAIL | #1 |
| **Pointer deref `*ptr`** | FAIL | #2 |
| **Sysvar assign `!p.multi = [...]`** | FAIL | #3 |
| **Arrow method `obj->Method`** | FAIL | #4 |
| **Class method `Class::Method`** | FAIL | #5 |

## Fix Priority

1. **#1 + #3** (struct/sysvar assignment) — single `assignment.left` change, fixes both
2. **#4** (arrow procedure calls) — new `method_procedure_call` statement
3. **#5** (class method definitions) — `qualified_name` for pro/function names
4. **#2** (pointer dereference) — needs conflict resolution with `*` multiply

Fixes 1-3 are straightforward grammar additions. Fix 4 requires care with `*` precedence ambiguity.

## Chunking Impact

For the imas-codex code discovery pipeline, these errors are **tolerable for chunking** — the parser still identifies `procedure_definition` and `function_definition` boundaries correctly even when the body has errors. The errors affect AST accuracy within function bodies but don't prevent top-level boundary detection, which is all `chunk_code` needs.

The fixes should still be made to enable future AST-based data access extraction (finding `mdsvalue` calls, tracking variable flow, etc.).
