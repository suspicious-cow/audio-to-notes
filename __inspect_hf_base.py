import ast
from pathlib import Path

env_root = Path("C:/Users/Zain_/anaconda3/envs/audio-notes-gpu/Lib/site-packages")
hf_path = env_root / "huggingface_hub" / "hub_mixin.py"
print(f"Reading {hf_path}")
source = hf_path.read_text(encoding="utf-8")
module = ast.parse(source)


class Visitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "_from_pretrained":
            args = [arg.arg for arg in node.args.args]
            kwonly = [arg.arg for arg in node.args.kwonlyargs]
            defaults = [None if d is None else ast.unparse(d) for d in node.args.defaults]
            kw_defaults = [None if d is None else ast.unparse(d) for d in node.args.kw_defaults]
            print(f"Found function {node.name}")
            print("args:", args)
            print("kwonly:", kwonly)
            print("defaults:", defaults)
            print("kw_defaults:", kw_defaults)
            segment = ast.get_source_segment(source, node)
            if segment:
                print("source snippet:\n" + '\n'.join(segment.split('\n')[:60]))


Visitor().visit(module)
