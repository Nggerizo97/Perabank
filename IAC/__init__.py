"""Infraestructura como código de PeraBank sobre el emulador AWS local.

La consola de Windows usa cp1252 por defecto y lanza UnicodeEncodeError al imprimir
los símbolos de estado (✔/✘). Se fuerza UTF-8 aquí porque este __init__ se ejecuta
antes que cualquier módulo del paquete.
"""
import sys

for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
