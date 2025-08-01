## Docy
You can add HTML tags inside doc to show in the documentation.

Rule:   
    Decorator: <p style="background-color:tomato;"><b> Decorator:</b></p>

## [sphinx](https://www.sphinx-doc.org/en/master/index.html) [01-08-25]
While reading the CPython documentation on multiprocessing, I noticed they use Sphinx for generating their documentation.
I checked PyTorch as well—same story.
It was surprising, since I had assumed that well-crafted documentation was a mark of a good programmer writing it alongside the code.

To be fair, Sphinx produces cleaner, more polished results than Docy. But my future plans align more closely with Docy’s philosophy. Tools like Sphinx often come with extra dependencies and, in some cases, are not fully open source—things I prefer to avoid.

My feelings about Sphinx are mixed. I'm a bit disappointed I didn’t recognize it earlier, but also somewhat relieved—I had genuinely believed the PyTorch docs were authored directly by developers during development.

So, I’ll stick with Docy. More than just a tool, Docy now reflects a mindset: that keeping good documentation is essential for helping developers perform better. What began as a simple library for generating docs has grown into a philosophy of programming I want to follow.

## Version 

- 01-08-25 [V] Add support for Python classes.

- 31-07-25 [V] Add unified method to add TODO, Adding keyword TODO to the function section and list all TODOs in the main page.

- 29-07-25 [V] Move CSS to separate folder

- 28-07-25 [V] Adding tags to automatically connect relevant functions from different implementations.
- 28-07-25 [V] Add functionality to import read me description for each src file and folder, and including it in web doc

- 19-07-25 [V] Add a link in each doc page to include that module easily.
   
