"""Run exact archived CPU postprocessors, with no Tele/model package imports."""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parent
SOURCE=json.loads((R/'NATIVE_DETERMINISTIC_CODE.json').read_text(encoding='utf8'))
FILES=SOURCE['files'];MODULES={}

def module(name):
 if name not in MODULES:
  namespace={'__name__':'frozen_native_'+name.removesuffix('.py')}
  exec(compile(FILES[name],SOURCE['source']+'/'+name,'exec'),namespace)
  MODULES[name]=namespace
 return MODULES[name]

def convert(raw,typ):
 if typ=='table':
  html=module('otsl2html.py')['convert_otsl_to_html'](raw)
  # Exact remove_useless_label function, extracted from archived source.
  n=next(x for x in ast.parse(FILES['__init__.py']).body if isinstance(x,ast.FunctionDef) and x.name=='remove_useless_label')
  namespace={};exec(compile(ast.Module(body=[n],type_ignores=[]),'archived_remove_useless_label','exec'),namespace)
  return namespace['remove_useless_label'](html)
 if typ=='text':return module('text_fix_dollar.py')['normalize_inline_math'](raw)
 if typ=='equation':
  namespace={}
  for fname in ['equation_remove_display_math_brackets.py','equation_double_dollar.py','equation_fix_begin_end.py','equation_unicode_latex.py','equation_left_right.py','equation_unbalanced_braces.py','equation_escape_latex_special_chars.py']:
   namespace.update(module(fname))
  n=next(x for x in ast.parse(FILES['__init__.py']).body if isinstance(x,ast.FunctionDef) and x.name=='_process_equation')
  exec(compile(ast.Module(body=[n],type_ignores=[]),'archived_process_equation','exec'),namespace)
  return namespace['_process_equation'](raw,False)
 return raw
