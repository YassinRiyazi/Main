import markdown

with open('doc/notes/notes_c.md', 'r') as f:
    c_notes = f.read()
with open('doc/notes/notes_python.md', 'r') as f:
    python_notes = f.read()

c_html = markdown.markdown(c_notes)
python_html = markdown.markdown(python_notes)

with open('Web/notes/notes_c.html', 'w') as f:
    f.write(c_html)
with open('Web/notes/notes_python.html', 'w') as f:
    f.write(python_html)