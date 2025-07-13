import os
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

def check_template_syntax():
    """Verificar sintaxis de todas las plantillas Jinja2"""
    template_dir = 'templates'
    
    if not os.path.exists(template_dir):
        print(f"❌ Directorio {template_dir} no encontrado")
        return False
    
    env = Environment(loader=FileSystemLoader(template_dir))
    
    errors = []
    success = []
    
    for filename in os.listdir(template_dir):
        if filename.endswith('.html'):
            try:
                template = env.get_template(filename)
                # Intentar compilar la plantilla
                template.render()
                success.append(filename)
                print(f"✅ {filename} - Sintaxis correcta")
            except TemplateSyntaxError as e:
                errors.append((filename, str(e)))
                print(f"❌ {filename} - Error: {e}")
            except Exception as e:
                # Otros errores como variables no definidas son esperados
                success.append(filename)
                print(f"✅ {filename} - Sintaxis correcta (variables no definidas esperadas)")
    
    print(f"\n📊 Resumen:")
    print(f"✅ Plantillas correctas: {len(success)}")
    print(f"❌ Plantillas con errores: {len(errors)}")
    
    if errors:
        print("\n🔧 Errores encontrados:")
        for filename, error in errors:
            print(f"  - {filename}: {error}")
    
    return len(errors) == 0

if __name__ == "__main__":
    print("🔍 Verificando sintaxis de plantillas Jinja2...\n")
    check_template_syntax()