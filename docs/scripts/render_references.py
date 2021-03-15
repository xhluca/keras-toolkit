from typing import NamedTuple, List
import inspect

from jinja2 import Template

import keras_toolkit as kt


type2str = {int: "int", float: "float", str: "str", bool: "bool", inspect._empty: "unspecified"}

default2str = {inspect._empty: "required", None: "optional"}


def preprocess_annot(annotation):
    annotation = type2str.get(annotation, str(annotation))

    return annotation.replace("typing.", "")


def preprocess_default(default):
    default = default2str.get(default, str(default))
    return default


class FunctionReference:
    def __init__(self, function):
        self.func = function

    @property
    def name(self):
        return self.func.__name__

    @property
    def modname(self):
        return inspect.getmodule(self.func).__name__.replace("keras_toolkit", "kt")

    @property
    def signature(self):
        return inspect.signature(self.func)

    @property
    def sig_desc(self):
        return self.modname + "." + self.name + str(self.signature)

    @property
    def doc(self):
        doc_template = Template(inspect.getdoc(self.func))
        kwargs = {
            "params": "| Parameter | Type | Default | Description |\n|-|-|-|-|"
        }
        for arg_name, param in self.signature.parameters.items():
            annot = preprocess_annot(param.annotation)
            default = preprocess_default(param.default)

            kwargs[arg_name] = f"| **{arg_name}** | *{annot}* | *{default}*"

        return doc_template.render(**kwargs)


ref = FunctionReference(kt.image.build_dataset)

with open("../templates/REFERENCES.md.jinja2") as f:
    template = Template(f.read())


out = template.render(func=ref)

with open('../REFERENCES.md', 'w') as f:
    f.write(out)