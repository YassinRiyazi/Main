# Why Docy?
This is Docky. A multipurpose documentation and simple word processor library. Now I need an HTML output, but it can also be expanded to TeX. 

For the reasons behind this library, read the following.

I’ve always had the desire to document my work. Initially, I used MS Word. I wrote a few notes and thoughts, but as my ideas and projects grew larger, I found myself limited by the tool. It was difficult to import images, organize the document in different ways, and make my work as dynamic as I wanted it to be. This was during my high school years. At this stage, I tried to build a WordPress website, but a lack of knowledge stopped me. Also, I have been amazed by x8.blogfa.com and other one witch provdied free VPN hpip.blogspot.com. But I knew how HTML could be powerful. FYI, WordPress was filtered. 

Later, I was introduced to LaTeX typesetting and [_Dr. Ali Mesforoush_](https://www.youtube.com/@DrMesforushAcademy/playlists). His free courses on MATLAB, LaTeX, and advocacy for GNU/free software had a profound impact on me. I sincerely hope Dr. Mesforoush thrives in this life and the next, if there is such a thing. I first learned the basics of LaTeX while working on a technical drawing book during the summer of my first year as a bachelor’s student. I vividly remember when my parents visited me in Tabriz. We stayed in a large house that my dad had reserved through his institution. I spent hours lying on the sofa, trying to create a simple LaTeX document in Persian—though, as you might know, language support for Persian, Arabic, and other non-Latin scripts was still quite limited back then. Surely it didn't help that I was not familiar with English as well.

The idea of having a document that could be processed like code was, and still is, incredibly exciting to me. However, there was a catch: you still had to follow the rule of sequential document structure, stack structure. You couldn’t add pages at arbitrary points in the document, like you could with a tree structure, to be a more concise HTML summary. This story takes place around 2018, a time when mighty ChatGPT wasn’t available, and ChatGPT-1 had just been introduced. Of course, it was possible to manipulate LaTeX to make documents more flexible, but it required more effort than I was willing to put in at the time.

Even if AI could have helped me achieve my goal of creating a flexible documentation system, I couldn’t—or to be more honest, I was too lazy—especially when it came to copying and pasting various elements, like code and additional information.

Later, during the final months of my Master’s program, I encountered a bizarre issue with MS Word. The header wouldn’t display correctly, and I couldn’t figure out whether it was due to the Persian language support or an error in the university thesis template, which conveniently they have  password-protected it. Either way, I had enough. I decided to rewrite the entire thesis in LaTeX. It took a solid month, with the revisions from my supervisor. I had 25 plots with similar descriptions, which would have been a nightmare to handle manually, but importing them into LaTeX using Python's f-string was a breeze.

Right now, it’s 2:30 AM, and I’m writing this weblog to remind myself why I’m investing so much time in WebDocy. I’ve rewritten this function over 40 F-ing times, always forgetting where the file is located.

 def get_subdirectories(root_dir, max_depth=2):

 directories = []

 for root, dirs, _ in sorted(os.walk(root_dir)):

 if root == root_dir:

 continue  # Skip the root directory itself

 depth = root[len(root_dir):].count(os.sep)

 if depth < max_depth:

 directories.append(root)

 else:

 del dirs[:]  # Stop descending further

 return directories

By forcing myself to document my work, keeping track of important pitfalls, and reminding myself why I added that specific if statement, I hope to do myself a favor in the long run.

## Word

It worth noting that MS Word is much better now, back then in 2012, if you had more that 80 pages, good luck with editing. 

## LaTex

Really great for PDF.

1. Also, I like the Label Ref style to link different hyperlinks. Which is accessible with \ Label: and \ Ref: style text inside notes.
2. Simple text editable source files.
3. High-quality graphics

## WebDocy

1. General-purpose documentation/diary.
2. Organizing
    1. toughs on my code
    2. Future plans (learning new skills)
    3. experiences
3. Doing compatibility checking between C, C++, and Python bindings. (Very basic compiler tasks)
4. Helping others to use my code.
5. Version control. GitHub
