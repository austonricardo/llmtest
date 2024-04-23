import sys
from odf.opendocument import load
from odf.text import ODFStyle, ParagraphStyle, STStyleStrikeout

def convert_to_txt(filename):
  """
  Converte um arquivo ODT para TXT com marcações de revogação.

  Args:
    filename: Nome do arquivo ODT.
  """

  doc = load(filename)
  text_body = doc.body

  # Criar um buffer para armazenar o conteúdo do texto
  txt_content = ""

  for paragraph in text_body.paragraphs:
    # Obter o estilo de parágrafo atual
    paragraph_style = paragraph.style

    # Verificar se o parágrafo possui a formatação riscada
    if paragraph_style.is_style(STStyleStrikeout):
      # Adicionar o texto do parágrafo com a marcação de revogação
      txt_content += paragraph.text + " **revogado**\n"
    else:
      # Adicionar o texto do parágrafo sem a marcação de revogação
      txt_content += paragraph.text + "\n"

  # Salvar o conteúdo do texto em um arquivo TXT
  with open(filename + ".txt", "w", encoding="utf-8") as f_out:
    f_out.write(txt_content)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Uso: convert_to_txt.py <arquivo.odt>")
    sys.exit(1)

  filename = sys.argv[1]
  convert_to_txt(filename)