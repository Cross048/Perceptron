import xml.etree.ElementTree as ET

from red_neuronal import RedNeuronal


def seleccionar_puerta():
    print("Seleccione la operación lógica a entrenar:")
    print("1. AND")
    print("2. OR")
    print("3. NAND")
    print("4. NOR")
    opcion = int(input("Ingrese el número de la opción: "))
    return opcion

def configurar_entradas(opcion):
    if opcion == 1:
        return [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 0, 0, 1], "AND"
    elif opcion == 2:
        return [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 1], "OR"
    elif opcion == 3:
        return [(0, 0), (0, 1), (1, 0), (1, 1)], [1, 1, 1, 0], "NAND"
    elif opcion == 4:
        return [(0, 0), (0, 1), (1, 0), (1, 1)], [1, 0, 0, 0], "NOR"
    else:
        print("Opción no válida.")
        return None, None, None

def guardar_pesos_en_xml(red, archivo_xml):
    root = ET.Element("red_neuronal")
    for idx, capa in enumerate(red.capas):
        capa_element = ET.SubElement(root, f"capa_{idx}")
        for j, neurona in enumerate(capa):
            neurona_element = ET.SubElement(capa_element, f"neurona_{j}")
            ET.SubElement(neurona_element, "pesos").text = ",".join(map(str, neurona.pesos))
            ET.SubElement(neurona_element, "sesgo").text = str(neurona.sesgo)
    tree = ET.ElementTree(root)
    tree.write(archivo_xml)
    print(f"\nPesos y sesgo guardados en {archivo_xml}")


def main():
    opcion = seleccionar_puerta()
    entradas, resultados_esperados, nombre_puerta = configurar_entradas(opcion)
    if not entradas:
        return

    # Cambiar la entrada para añadir la última capa de salida
    neuronas_por_capa_oculta = input("Ingrese el número de neuronas por capa oculta (separadas por coma): ")

    # Procesamos la entrada y aseguramos que la última capa sea siempre 1
    estructura = [2]  # La capa de entrada con 2 neuronas
    if neuronas_por_capa_oculta:
        capas_ocultas = [int(n) for n in neuronas_por_capa_oculta.split(",")]
        estructura.extend(capas_ocultas)
    estructura.append(1)  # La capa de salida siempre con 1 neurona

    red = RedNeuronal(estructura)

    tasa_aprendizaje = 0.1
    ciclos = 10000

    # Entrenar la red
    red.entrenar(entradas, resultados_esperados, tasa_aprendizaje, ciclos)

    # Guardar los pesos en un archivo XML
    archivo_xml = f"pesos_{nombre_puerta}.xml"
    guardar_pesos_en_xml(red, archivo_xml)

    # Probar la red
    print(f"\nPrueba de la red entrenada para {nombre_puerta}:")
    for entrada in entradas:
        resultado = red.pasada_adelante(entrada)
        print(f"Entrada: {entrada}, Salida: {resultado[0]:.5f}")


if __name__ == "__main__":
    main()