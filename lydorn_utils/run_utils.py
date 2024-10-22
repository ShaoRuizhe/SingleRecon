#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from jsmin import jsmin
import json

from . import print_utils
from . import python_utils

# Stolen from Docker:
NAME_SET = set([
    # Muhammad ibn Jābir al-Ḥarrānī al-Battānī was a founding father of astronomy. https://en.wikipedia.org/wiki/Mu%E1%B8%A5ammad_ibn_J%C4%81bir_al-%E1%B8%A4arr%C4%81n%C4%AB_al-Batt%C4%81n%C4%AB
    "albattani",

    # Frances E. Allen, became the first female IBM Fellow in 1989. In 2006, she became the first female recipient of the ACM's Turing Award. https://en.wikipedia.org/wiki/Frances_E._Allen
    "allen",

    # June Almeida - Scottish virologist who took the first pictures of the rubella virus - https://en.wikipedia.org/wiki/June_Almeida
    "almeida",

    # Maria Gaetana Agnesi - Italian mathematician, philosopher, theologian and humanitarian. She was the first woman to write a mathematics handbook and the first woman appointed as a Mathematics Professor at a University. https://en.wikipedia.org/wiki/Maria_Gaetana_Agnesi
    "agnesi",

    # Archimedes was a physicist, engineer and mathematician who invented too many things to list them here. https://en.wikipedia.org/wiki/Archimedes
    "archimedes",

    # Maria Ardinghelli - Italian translator, mathematician and physicist - https://en.wikipedia.org/wiki/Maria_Ardinghelli
    "ardinghelli",

    # Aryabhata - Ancient Indian mathematician-astronomer during 476-550 CE https://en.wikipedia.org/wiki/Aryabhata
    "aryabhata",

    # Wanda Austin - Wanda Austin is the President and CEO of The Aerospace Corporation, a leading architect for the US security space programs. https://en.wikipedia.org/wiki/Wanda_Austin
    "austin",

    # Charles Babbage invented the concept of a programmable computer. https://en.wikipedia.org/wiki/Charles_Babbage.
    "babbage",

    # Stefan Banach - Polish mathematician, was one of the founders of modern functional analysis. https://en.wikipedia.org/wiki/Stefan_Banach
    "banach",

    # John Bardeen co-invented the transistor - https://en.wikipedia.org/wiki/John_Bardeen
    "bardeen",

    # Jean Bartik, born Betty Jean Jennings, was one of the original programmers for the ENIAC computer. https://en.wikipedia.org/wiki/Jean_Bartik
    "bartik",

    # Laura Bassi, the world's first female professor https://en.wikipedia.org/wiki/Laura_Bassi
    "bassi",

    # Hugh Beaver, British engineer, founder of the Guinness Book of World Records https://en.wikipedia.org/wiki/Hugh_Beaver
    "beaver",

    # Alexander Graham Bell - an eminent Scottish-born scientist, inventor, engineer and innovator who is credited with inventing the first practical telephone - https://en.wikipedia.org/wiki/Alexander_Graham_Bell
    "bell",

    # Karl Friedrich Benz - a German automobile engineer. Inventor of the first practical motorcar. https://en.wikipedia.org/wiki/Karl_Benz
    "benz",

    # Homi J Bhabha - was an Indian nuclear physicist, founding director, and professor of physics at the Tata Institute of Fundamental Research. Colloquially known as "father of Indian nuclear programme"- https://en.wikipedia.org/wiki/Homi_J._Bhabha
    "bhabha",

    # Bhaskara II - Ancient Indian mathematician-astronomer whose work on calculus predates Newton and Leibniz by over half a millennium - https://en.wikipedia.org/wiki/Bh%C4%81skara_II#Calculus
    "bhaskara",

    # Elizabeth Blackwell - American doctor and first American woman to receive a medical degree - https://en.wikipedia.org/wiki/Elizabeth_Blackwell
    "blackwell",

    # Niels Bohr is the father of quantum theory. https://en.wikipedia.org/wiki/Niels_Bohr.
    "bohr",

    # Kathleen Booth, she's credited with writing the first assembly language. https://en.wikipedia.org/wiki/Kathleen_Booth
    "booth",

    # Anita Borg - Anita Borg was the founding director of the Institute for Women and Technology (IWT). https://en.wikipedia.org/wiki/Anita_Borg
    "borg",

    # Satyendra Nath Bose - He provided the foundation for Bose–Einstein statistics and the theory of the Bose–Einstein condensate. - https://en.wikipedia.org/wiki/Satyendra_Nath_Bose
    "bose",

    # Evelyn Boyd Granville - She was one of the first African-American woman to receive a Ph.D. in mathematics; she earned it in 1949 from Yale University. https://en.wikipedia.org/wiki/Evelyn_Boyd_Granville
    "boyd",

    # Brahmagupta - Ancient Indian mathematician during 598-670 CE who gave rules to compute with zero - https://en.wikipedia.org/wiki/Brahmagupta#Zero
    "brahmagupta",

    # Walter Houser Brattain co-invented the transistor - https://en.wikipedia.org/wiki/Walter_Houser_Brattain
    "brattain",

    # Emmett Brown invented time travel. https://en.wikipedia.org/wiki/Emmett_Brown (thanks Brian Goff)
    "brown",

    # Rachel Carson - American marine biologist and conservationist, her book Silent Spring and other writings are credited with advancing the global environmental movement. https://en.wikipedia.org/wiki/Rachel_Carson
    "carson",

    # Subrahmanyan Chandrasekhar - Astrophysicist known for his mathematical theory on different stages and evolution in structures of the stars. He has won nobel prize for physics - https://en.wikipedia.org/wiki/Subrahmanyan_Chandrasekhar
    "chandrasekhar",

    # Sergey Alexeyevich Chaplygin (Russian: Серге́й Алексе́евич Чаплы́гин; April 5, 1869 – October 8, 1942) was a Russian and Soviet physicist, mathematician, and mechanical engineer. He is known for mathematical formulas such as Chaplygin's equation and for a hypothetical substance in cosmology called Chaplygin gas, named after him. https://en.wikipedia.org/wiki/Sergey_Chaplygin
    "chaplygin",

    # Asima Chatterjee was an indian organic chemist noted for her research on vinca alkaloids, development of drugs for treatment of epilepsy and malaria - https://en.wikipedia.org/wiki/Asima_Chatterjee
    "chatterjee",

    # Pafnuty Chebyshev - Russian mathematitian. He is known fo his works on probability, statistics, mechanics, analytical geometry and number theory https://en.wikipedia.org/wiki/Pafnuty_Chebyshev
    "chebyshev",

    # Claude Shannon - The father of information theory and founder of digital circuit design theory. (https://en.wikipedia.org/wiki/Claude_Shannon)
    "shannon",

    # Joan Clarke - Bletchley Park code breaker during the Second World War who pioneered techniques that remained top secret for decades. Also an accomplished numismatist https://en.wikipedia.org/wiki/Joan_Clarke
    "clarke",

    # Jane Colden - American botanist widely considered the first female American botanist - https://en.wikipedia.org/wiki/Jane_Colden
    "colden",

    # Gerty Theresa Cori - American biochemist who became the third woman—and first American woman—to win a Nobel Prize in science, and the first woman to be awarded the Nobel Prize in Physiology or Medicine. Cori was born in Prague. https://en.wikipedia.org/wiki/Gerty_Cori
    "cori",

    # Seymour Roger Cray was an American electrical engineer and supercomputer architect who designed a series of computers that were the fastest in the world for decades. https://en.wikipedia.org/wiki/Seymour_Cray
    "cray",

    # This entry reflects a husband and wife team who worked together:
    # Joan Curran was a Welsh scientist who developed radar and invented chaff, a radar countermeasure. https://en.wikipedia.org/wiki/Joan_Curran
    # Samuel Curran was an Irish physicist who worked alongside his wife during WWII and invented the proximity fuse. https://en.wikipedia.org/wiki/Samuel_Curran
    "curran",

    # Marie Curie discovered radioactivity. https://en.wikipedia.org/wiki/Marie_Curie.
    "curie",

    # Charles Darwin established the principles of natural evolution. https://en.wikipedia.org/wiki/Charles_Darwin.
    "darwin",

    # Leonardo Da Vinci invented too many things to list here. https://en.wikipedia.org/wiki/Leonardo_da_Vinci.
    "davinci",

    # Edsger Wybe Dijkstra was a Dutch computer scientist and mathematical scientist. https://en.wikipedia.org/wiki/Edsger_W._Dijkstra.
    "dijkstra",

    # Donna Dubinsky - played an integral role in the development of personal digital assistants (PDAs) serving as CEO of Palm, Inc. and co-founding Handspring. https://en.wikipedia.org/wiki/Donna_Dubinsky
    "dubinsky",

    # Annie Easley - She was a leading member of the team which developed software for the Centaur rocket stage and one of the first African-Americans in her field. https://en.wikipedia.org/wiki/Annie_Easley
    "easley",

    # Thomas Alva Edison, prolific inventor https://en.wikipedia.org/wiki/Thomas_Edison
    "edison",

    # Albert Einstein invented the general theory of relativity. https://en.wikipedia.org/wiki/Albert_Einstein
    "einstein",

    # Gertrude Elion - American biochemist, pharmacologist and the 1988 recipient of the Nobel Prize in Medicine - https://en.wikipedia.org/wiki/Gertrude_Elion
    "elion",

    # Alexandra Asanovna Elbakyan (Russian: Алекса́ндра Аса́новна Элбакя́н) is a Kazakhstani graduate student, computer programmer, internet pirate in hiding, and the creator of the site Sci-Hub. Nature has listed her in 2016 in the top ten people that mattered in science, and Ars Technica has compared her to Aaron Swartz. - https://en.wikipedia.org/wiki/Alexandra_Elbakyan
    "elbakyan",

    # Douglas Engelbart gave the mother of all demos: https://en.wikipedia.org/wiki/Douglas_Engelbart
    "engelbart",

    # Euclid invented geometry. https://en.wikipedia.org/wiki/Euclid
    "euclid",

    # Leonhard Euler invented large parts of modern mathematics. https://de.wikipedia.org/wiki/Leonhard_Euler
    "euler",

    # Pierre de Fermat pioneered several aspects of modern mathematics. https://en.wikipedia.org/wiki/Pierre_de_Fermat
    "fermat",

    # Enrico Fermi invented the first nuclear reactor. https://en.wikipedia.org/wiki/Enrico_Fermi.
    "fermi",

    # Richard Feynman was a key contributor to quantum mechanics and particle physics. https://en.wikipedia.org/wiki/Richard_Feynman
    "feynman",

    # Benjamin Franklin is famous for his experiments in electricity and the invention of the lightning rod.
    "franklin",

    # Galileo was a founding father of modern astronomy, and faced politics and obscurantism to establish scientific truth.  https://en.wikipedia.org/wiki/Galileo_Galilei
    "galileo",

    # William Henry "Bill" Gates III is an American business magnate, philanthropist, investor, computer programmer, and inventor. https://en.wikipedia.org/wiki/Bill_Gates
    "gates",

    # Adele Goldberg, was one of the designers and developers of the Smalltalk language. https://en.wikipedia.org/wiki/Adele_Goldberg_(computer_scientist)
    "goldberg",

    # Adele Goldstine, born Adele Katz, wrote the complete technical description for the first electronic digital computer, ENIAC. https://en.wikipedia.org/wiki/Adele_Goldstine
    "goldstine",

    # Shafi Goldwasser is a computer scientist known for creating theoretical foundations of modern cryptography. Winner of 2012 ACM Turing Award. https://en.wikipedia.org/wiki/Shafi_Goldwasser
    "goldwasser",

    # James Golick, all around gangster.
    "golick",

    # Jane Goodall - British primatologist, ethologist, and anthropologist who is considered to be the world's foremost expert on chimpanzees - https://en.wikipedia.org/wiki/Jane_Goodall
    "goodall",

    # Lois Haibt - American computer scientist, part of the team at IBM that developed FORTRAN - https://en.wikipedia.org/wiki/Lois_Haibt
    "haibt",

    # Margaret Hamilton - Director of the Software Engineering Division of the MIT Instrumentation Laboratory, which developed on-board flight software for the Apollo space program. https://en.wikipedia.org/wiki/Margaret_Hamilton_(scientist)
    "hamilton",

    # Stephen Hawking pioneered the field of cosmology by combining general relativity and quantum mechanics. https://en.wikipedia.org/wiki/Stephen_Hawking
    "hawking",

    # Werner Heisenberg was a founding father of quantum mechanics. https://en.wikipedia.org/wiki/Werner_Heisenberg
    "heisenberg",

    # Grete Hermann was a German philosopher noted for her philosophical work on the foundations of quantum mechanics. https://en.wikipedia.org/wiki/Grete_Hermann
    "hermann",

    # Jaroslav Heyrovský was the inventor of the polarographic method, father of the electroanalytical method, and recipient of the Nobel Prize in 1959. His main field of work was polarography. https://en.wikipedia.org/wiki/Jaroslav_Heyrovsk%C3%BD
    "heyrovsky",

    # Dorothy Hodgkin was a British biochemist, credited with the development of protein crystallography. She was awarded the Nobel Prize in Chemistry in 1964. https://en.wikipedia.org/wiki/Dorothy_Hodgkin
    "hodgkin",

    # Erna Schneider Hoover revolutionized modern communication by inventing a computerized telephone switching method. https://en.wikipedia.org/wiki/Erna_Schneider_Hoover
    "hoover",

    # Grace Hopper developed the first compiler for a computer programming language and  is credited with popularizing the term "debugging" for fixing computer glitches. https://en.wikipedia.org/wiki/Grace_Hopper
    "hopper",

    # Frances Hugle, she was an American scientist, engineer, and inventor who contributed to the understanding of semiconductors, integrated circuitry, and the unique electrical principles of microscopic materials. https://en.wikipedia.org/wiki/Frances_Hugle
    "hugle",

    # Hypatia - Greek Alexandrine Neoplatonist philosopher in Egypt who was one of the earliest mothers of mathematics - https://en.wikipedia.org/wiki/Hypatia
    "hypatia",

    # Mary Jackson, American mathematician and aerospace engineer who earned the highest title within NASA's engineering department - https://en.wikipedia.org/wiki/Mary_Jackson_(engineer)
    "jackson",

    # Yeong-Sil Jang was a Korean scientist and astronomer during the Joseon Dynasty; he invented the first metal printing press and water gauge. https://en.wikipedia.org/wiki/Jang_Yeong-sil
    "jang",

    # Betty Jennings - one of the original programmers of the ENIAC. https://en.wikipedia.org/wiki/ENIAC - https://en.wikipedia.org/wiki/Jean_Bartik
    "jennings",

    # Mary Lou Jepsen, was the founder and chief technology officer of One Laptop Per Child (OLPC), and the founder of Pixel Qi. https://en.wikipedia.org/wiki/Mary_Lou_Jepsen
    "jepsen",

    # Katherine Coleman Goble Johnson - American physicist and mathematician contributed to the NASA. https://en.wikipedia.org/wiki/Katherine_Johnson
    "johnson",

    # Irène Joliot-Curie - French scientist who was awarded the Nobel Prize for Chemistry in 1935. Daughter of Marie and Pierre Curie. https://en.wikipedia.org/wiki/Ir%C3%A8ne_Joliot-Curie
    "joliot",

    # Karen Spärck Jones came up with the concept of inverse document frequency, which is used in most search engines today. https://en.wikipedia.org/wiki/Karen_Sp%C3%A4rck_Jones
    "jones",

    # A. P. J. Abdul Kalam - is an Indian scientist aka Missile Man of India for his work on the development of ballistic missile and launch vehicle technology - https://en.wikipedia.org/wiki/A._P._J._Abdul_Kalam
    "kalam",

    # Sergey Petrovich Kapitsa (Russian: Серге́й Петро́вич Капи́ца; 14 February 1928 – 14 August 2012) was a Russian physicist and demographer. He was best known as host of the popular and long-running Russian scientific TV show, Evident, but Incredible. His father was the Nobel laureate Soviet-era physicist Pyotr Kapitsa, and his brother was the geographer and Antarctic explorer Andrey Kapitsa. - https://en.wikipedia.org/wiki/Sergey_Kapitsa
    "kapitsa",

    # Susan Kare, created the icons and many of the interface elements for the original Apple Macintosh in the 1980s, and was an original employee of NeXT, working as the Creative Director. https://en.wikipedia.org/wiki/Susan_Kare
    "kare",

    # Mstislav Keldysh - a Soviet scientist in the field of mathematics and mechanics, academician of the USSR Academy of Sciences (1946), President of the USSR Academy of Sciences (1961–1975), three times Hero of Socialist Labor (1956, 1961, 1971), fellow of the Royal Society of Edinburgh (1968). https://en.wikipedia.org/wiki/Mstislav_Keldysh
    "keldysh",

    # Mary Kenneth Keller, Sister Mary Kenneth Keller became the first American woman to earn a PhD in Computer Science in 1965. https://en.wikipedia.org/wiki/Mary_Kenneth_Keller
    "keller",

    # Johannes Kepler, German astronomer known for his three laws of planetary motion - https://en.wikipedia.org/wiki/Johannes_Kepler
    "kepler",

    # Har Gobind Khorana - Indian-American biochemist who shared the 1968 Nobel Prize for Physiology - https://en.wikipedia.org/wiki/Har_Gobind_Khorana
    "khorana",

    # Jack Kilby invented silicone integrated circuits and gave Silicon Valley its name. - https://en.wikipedia.org/wiki/Jack_Kilby
    "kilby",

    # Maria Kirch - German astronomer and first woman to discover a comet - https://en.wikipedia.org/wiki/Maria_Margarethe_Kirch
    "kirch",

    # Donald Knuth - American computer scientist, author of "The Art of Computer Programming" and creator of the TeX typesetting system. https://en.wikipedia.org/wiki/Donald_Knuth
    "knuth",

    # Sophie Kowalevski - Russian mathematician responsible for important original contributions to analysis, differential equations and mechanics - https://en.wikipedia.org/wiki/Sofia_Kovalevskaya
    "kowalevski",

    # Marie-Jeanne de Lalande - French astronomer, mathematician and cataloguer of stars - https://en.wikipedia.org/wiki/Marie-Jeanne_de_Lalande
    "lalande",

    # Hedy Lamarr - Actress and inventor. The principles of her work are now incorporated into modern Wi-Fi, CDMA and Bluetooth technology. https://en.wikipedia.org/wiki/Hedy_Lamarr
    "lamarr",

    # Leslie B. Lamport - American computer scientist. Lamport is best known for his seminal work in distributed systems and was the winner of the 2013 Turing Award. https://en.wikipedia.org/wiki/Leslie_Lamport
    "lamport",

    # Mary Leakey - British paleoanthropologist who discovered the first fossilized Proconsul skull - https://en.wikipedia.org/wiki/Mary_Leakey
    "leakey",

    # Henrietta Swan Leavitt - she was an American astronomer who discovered the relation between the luminosity and the period of Cepheid variable stars. https://en.wikipedia.org/wiki/Henrietta_Swan_Leavitt
    "leavitt",

    # Daniel Lewin -  Mathematician, Akamai co-founder, soldier, 9/11 victim-- Developed optimization techniques for routing traffic on the internet. Died attempting to stop the 9-11 hijackers. https://en.wikipedia.org/wiki/Daniel_Lewin
    "lewin",

    # Ruth Lichterman - one of the original programmers of the ENIAC. https://en.wikipedia.org/wiki/ENIAC - https://en.wikipedia.org/wiki/Ruth_Teitelbaum
    "lichterman",

    # Barbara Liskov - co-developed the Liskov substitution principle. Liskov was also the winner of the Turing Prize in 2008. - https://en.wikipedia.org/wiki/Barbara_Liskov
    "liskov",

    # Ada Lovelace invented the first algorithm. https://en.wikipedia.org/wiki/Ada_Lovelace (thanks James Turnbull)
    "lovelace",

    # Auguste and Louis Lumière - the first filmmakers in history - https://en.wikipedia.org/wiki/Auguste_and_Louis_Lumi%C3%A8re
    "lumiere",

    # Mahavira - Ancient Indian mathematician during 9th century AD who discovered basic algebraic identities - https://en.wikipedia.org/wiki/Mah%C4%81v%C4%ABra_(mathematician)
    "mahavira",

    # Maria Mayer - American theoretical physicist and Nobel laureate in Physics for proposing the nuclear shell model of the atomic nucleus - https://en.wikipedia.org/wiki/Maria_Mayer
    "mayer",

    # John McCarthy invented LISP: https://en.wikipedia.org/wiki/John_McCarthy_(computer_scientist)
    "mccarthy",

    # Barbara McClintock - a distinguished American cytogeneticist, 1983 Nobel Laureate in Physiology or Medicine for discovering transposons. https://en.wikipedia.org/wiki/Barbara_McClintock
    "mcclintock",

    # Malcolm McLean invented the modern shipping container: https://en.wikipedia.org/wiki/Malcom_McLean
    "mclean",

    # Kay McNulty - one of the original programmers of the ENIAC. https://en.wikipedia.org/wiki/ENIAC - https://en.wikipedia.org/wiki/Kathleen_Antonelli
    "mcnulty",

    # Dmitri Mendeleev - a chemist and inventor. He formulated the Periodic Law, created a farsighted version of the periodic table of elements, and used it to correct the properties of some already discovered elements and also to predict the properties of eight elements yet to be discovered. https://en.wikipedia.org/wiki/Dmitri_Mendeleev
    "mendeleev",

    # Lise Meitner - Austrian/Swedish physicist who was involved in the discovery of nuclear fission. The element meitnerium is named after her - https://en.wikipedia.org/wiki/Lise_Meitner
    "meitner",

    # Carla Meninsky, was the game designer and programmer for Atari 2600 games Dodge 'Em and Warlords. https://en.wikipedia.org/wiki/Carla_Meninsky
    "meninsky",

    # Johanna Mestorf - German prehistoric archaeologist and first female museum director in Germany - https://en.wikipedia.org/wiki/Johanna_Mestorf
    "mestorf",

    # Marvin Minsky - Pioneer in Artificial Intelligence, co-founder of the MIT's AI Lab, won the Turing Award in 1969. https://en.wikipedia.org/wiki/Marvin_Minsky
    "minsky",

    # Maryam Mirzakhani - an Iranian mathematician and the first woman to win the Fields Medal. https://en.wikipedia.org/wiki/Maryam_Mirzakhani
    "mirzakhani",

    # Samuel Morse - contributed to the invention of a single-wire telegraph system based on European telegraphs and was a co-developer of the Morse code - https://en.wikipedia.org/wiki/Samuel_Morse
    "morse",

    # Ian Murdock - founder of the Debian project - https://en.wikipedia.org/wiki/Ian_Murdock
    "murdock",

    # John von Neumann - todays computer architectures are based on the von Neumann architecture. https://en.wikipedia.org/wiki/Von_Neumann_architecture
    "neumann",

    # Isaac Newton invented classic mechanics and modern optics. https://en.wikipedia.org/wiki/Isaac_Newton
    "newton",

    # Florence Nightingale, more prominently known as a nurse, was also the first female member of the Royal Statistical Society and a pioneer in statistical graphics https://en.wikipedia.org/wiki/Florence_Nightingale#Statistics_and_sanitary_reform
    "nightingale",

    # Alfred Nobel - a Swedish chemist, engineer, innovator, and armaments manufacturer (inventor of dynamite) - https://en.wikipedia.org/wiki/Alfred_Nobel
    "nobel",

    # Emmy Noether, German mathematician. Noether's Theorem is named after her. https://en.wikipedia.org/wiki/Emmy_Noether
    "noether",

    # Poppy Northcutt. Poppy Northcutt was the first woman to work as part of NASA’s Mission Control. http://www.businessinsider.com/poppy-northcutt-helped-apollo-astronauts-2014-12?op=1
    "northcutt",

    # Robert Noyce invented silicone integrated circuits and gave Silicon Valley its name. - https://en.wikipedia.org/wiki/Robert_Noyce
    "noyce",

    # Panini - Ancient Indian linguist and grammarian from 4th century CE who worked on the world's first formal system - https://en.wikipedia.org/wiki/P%C4%81%E1%B9%87ini#Comparison_with_modern_formal_systems
    "panini",

    # Ambroise Pare invented modern surgery. https://en.wikipedia.org/wiki/Ambroise_Par%C3%A9
    "pare",

    # Louis Pasteur discovered vaccination, fermentation and pasteurization. https://en.wikipedia.org/wiki/Louis_Pasteur.
    "pasteur",

    # Cecilia Payne-Gaposchkin was an astronomer and astrophysicist who, in 1925, proposed in her Ph.D. thesis an explanation for the composition of stars in terms of the relative abundances of hydrogen and helium. https://en.wikipedia.org/wiki/Cecilia_Payne-Gaposchkin
    "payne",

    # Radia Perlman is a software designer and network engineer and most famous for her invention of the spanning-tree protocol (STP). https://en.wikipedia.org/wiki/Radia_Perlman
    "perlman",

    # Rob Pike was a key contributor to Unix, Plan 9, the X graphic system, utf-8, and the Go programming language. https://en.wikipedia.org/wiki/Rob_Pike
    "pike",

    # Henri Poincaré made fundamental contributions in several fields of mathematics. https://en.wikipedia.org/wiki/Henri_Poincar%C3%A9
    "poincare",

    # Laura Poitras is a director and producer whose work, made possible by open source crypto tools, advances the causes of truth and freedom of information by reporting disclosures by whistleblowers such as Edward Snowden. https://en.wikipedia.org/wiki/Laura_Poitras
    "poitras",

    # Tat’yana Avenirovna Proskuriakova (Russian: Татья́на Авени́ровна Проскуряко́ва) (January 23 [O.S. January 10] 1909 – August 30, 1985) was a Russian-American Mayanist scholar and archaeologist who contributed significantly to the deciphering of Maya hieroglyphs, the writing system of the pre-Columbian Maya civilization of Mesoamerica. https://en.wikipedia.org/wiki/Tatiana_Proskouriakoff
    "proskuriakova",

    # Claudius Ptolemy - a Greco-Egyptian writer of Alexandria, known as a mathematician, astronomer, geographer, astrologer, and poet of a single epigram in the Greek Anthology - https://en.wikipedia.org/wiki/Ptolemy
    "ptolemy",

    # C. V. Raman - Indian physicist who won the Nobel Prize in 1930 for proposing the Raman effect. - https://en.wikipedia.org/wiki/C._V._Raman
    "raman",

    # Srinivasa Ramanujan - Indian mathematician and autodidact who made extraordinary contributions to mathematical analysis, number theory, infinite series, and continued fractions. - https://en.wikipedia.org/wiki/Srinivasa_Ramanujan
    "ramanujan",

    # Sally Kristen Ride was an American physicist and astronaut. She was the first American woman in space, and the youngest American astronaut. https://en.wikipedia.org/wiki/Sally_Ride
    "ride",

    # Rita Levi-Montalcini - Won Nobel Prize in Physiology or Medicine jointly with colleague Stanley Cohen for the discovery of nerve growth factor (https://en.wikipedia.org/wiki/Rita_Levi-Montalcini)
    "montalcini",

    # Dennis Ritchie - co-creator of UNIX and the C programming language. - https://en.wikipedia.org/wiki/Dennis_Ritchie
    "ritchie",

    # Wilhelm Conrad Röntgen - German physicist who was awarded the first Nobel Prize in Physics in 1901 for the discovery of X-rays (Röntgen rays). https://en.wikipedia.org/wiki/Wilhelm_R%C3%B6ntgen
    "roentgen",

    # Rosalind Franklin - British biophysicist and X-ray crystallographer whose research was critical to the understanding of DNA - https://en.wikipedia.org/wiki/Rosalind_Franklin
    "rosalind",

    # Meghnad Saha - Indian astrophysicist best known for his development of the Saha equation, used to describe chemical and physical conditions in stars - https://en.wikipedia.org/wiki/Meghnad_Saha
    "saha",

    # Jean E. Sammet developed FORMAC, the first widely used computer language for symbolic manipulation of mathematical formulas. https://en.wikipedia.org/wiki/Jean_E._Sammet
    "sammet",

    # Carol Shaw - Originally an Atari employee, Carol Shaw is said to be the first female video game designer. https://en.wikipedia.org/wiki/Carol_Shaw_(video_game_designer)
    "shaw",

    # Dame Stephanie "Steve" Shirley - Founded a software company in 1962 employing women working from home. https://en.wikipedia.org/wiki/Steve_Shirley
    "shirley",

    # William Shockley co-invented the transistor - https://en.wikipedia.org/wiki/William_Shockley
    "shockley",

    # Françoise Barré-Sinoussi - French virologist and Nobel Prize Laureate in Physiology or Medicine; her work was fundamental in identifying HIV as the cause of AIDS. https://en.wikipedia.org/wiki/Fran%C3%A7oise_Barr%C3%A9-Sinoussi
    "sinoussi",

    # Betty Snyder - one of the original programmers of the ENIAC. https://en.wikipedia.org/wiki/ENIAC - https://en.wikipedia.org/wiki/Betty_Holberton
    "snyder",

    # Frances Spence - one of the original programmers of the ENIAC. https://en.wikipedia.org/wiki/ENIAC - https://en.wikipedia.org/wiki/Frances_Spence
    "spence",

    # Richard Matthew Stallman - the founder of the Free Software movement, the GNU project, the Free Software Foundation, and the League for Programming Freedom. He also invented the concept of copyleft to protect the ideals of this movement, and enshrined this concept in the widely-used GPL (General Public License) for software. https://en.wikiquote.org/wiki/Richard_Stallman
    "stallman",

    # Lina Solomonovna Stern (or Shtern; Russian: Лина Соломоновна Штерн; 26 August 1878 – 7 March 1968) was a Soviet biochemist, physiologist and humanist whose medical discoveries saved thousands of lives at the fronts of World War II. She is best known for her pioneering work on blood–brain barrier, which she described as hemato-encephalic barrier in 1921. https://en.wikipedia.org/wiki/Lina_Stern
    "shtern",

    # Michael Stonebraker is a database research pioneer and architect of Ingres, Postgres, VoltDB and SciDB. Winner of 2014 ACM Turing Award. https://en.wikipedia.org/wiki/Michael_Stonebraker
    "stonebraker",

    # Janese Swanson (with others) developed the first of the Carmen Sandiego games. She went on to found Girl Tech. https://en.wikipedia.org/wiki/Janese_Swanson
    "swanson",

    # Aaron Swartz was influential in creating RSS, Markdown, Creative Commons, Reddit, and much of the internet as we know it today. He was devoted to freedom of information on the web. https://en.wikiquote.org/wiki/Aaron_Swartz
    "swartz",

    # Bertha Swirles was a theoretical physicist who made a number of contributions to early quantum theory. https://en.wikipedia.org/wiki/Bertha_Swirles
    "swirles",

    # Valentina Tereshkova is a russian engineer, cosmonaut and politician. She was the first woman flying to space in 1963. In 2013, at the age of 76, she offered to go on a one-way mission to mars. https://en.wikipedia.org/wiki/Valentina_Tereshkova
    "tereshkova",

    # Nikola Tesla invented the AC electric system and every gadget ever used by a James Bond villain. https://en.wikipedia.org/wiki/Nikola_Tesla
    "tesla",

    # Ken Thompson - co-creator of UNIX and the C programming language - https://en.wikipedia.org/wiki/Ken_Thompson
    "thompson",

    # Linus Torvalds invented Linux and Git. https://en.wikipedia.org/wiki/Linus_Torvalds
    "torvalds",

    # Alan Turing was a founding father of computer science. https://en.wikipedia.org/wiki/Alan_Turing.
    "turing",

    # Varahamihira - Ancient Indian mathematician who discovered trigonometric formulae during 505-587 CE - https://en.wikipedia.org/wiki/Var%C4%81hamihira#Contributions
    "varahamihira",

    # Dorothy Vaughan was a NASA mathematician and computer programmer on the SCOUT launch vehicle program that put America's first satellites into space - https://en.wikipedia.org/wiki/Dorothy_Vaughan
    "vaughan",

    # Sir Mokshagundam Visvesvaraya - is a notable Indian engineer.  He is a recipient of the Indian Republic's highest honour, the Bharat Ratna, in 1955. On his birthday, 15 September is celebrated as Engineer's Day in India in his memory - https://en.wikipedia.org/wiki/Visvesvaraya
    "visvesvaraya",

    # Christiane Nüsslein-Volhard - German biologist, won Nobel Prize in Physiology or Medicine in 1995 for research on the genetic control of embryonic development. https://en.wikipedia.org/wiki/Christiane_N%C3%BCsslein-Volhard
    "volhard",

    # Cédric Villani - French mathematician, won Fields Medal, Fermat Prize and Poincaré Price for his work in differential geometry and statistical mechanics. https://en.wikipedia.org/wiki/C%C3%A9dric_Villani
    "villani",

    # Marlyn Wescoff - one of the original programmers of the ENIAC. https://en.wikipedia.org/wiki/ENIAC - https://en.wikipedia.org/wiki/Marlyn_Meltzer
    "wescoff",

    # Andrew Wiles - Notable British mathematician who proved the enigmatic Fermat's Last Theorem - https://en.wikipedia.org/wiki/Andrew_Wiles
    "wiles",

    # Roberta Williams, did pioneering work in graphical adventure games for personal computers, particularly the King's Quest series. https://en.wikipedia.org/wiki/Roberta_Williams
    "williams",

    # Sophie Wilson designed the first Acorn Micro-Computer and the instruction set for ARM processors. https://en.wikipedia.org/wiki/Sophie_Wilson
    "wilson",

    # Jeannette Wing - co-developed the Liskov substitution principle. - https://en.wikipedia.org/wiki/Jeannette_Wing
    "wing",

    # Steve Wozniak invented the Apple I and Apple II. https://en.wikipedia.org/wiki/Steve_Wozniak
    "wozniak",

    # The Wright brothers, Orville and Wilbur - credited with inventing and building the world's first successful airplane and making the first controlled, powered and sustained heavier-than-air human flight - https://en.wikipedia.org/wiki/Wright_brothers
    "wright",

    # Rosalyn Sussman Yalow - Rosalyn Sussman Yalow was an American medical physicist, and a co-winner of the 1977 Nobel Prize in Physiology or Medicine for development of the radioimmunoassay technique. https://en.wikipedia.org/wiki/Rosalyn_Sussman_Yalow
    "yalow",

    # Ada Yonath - an Israeli crystallographer, the first woman from the Middle East to win a Nobel prize in the sciences. https://en.wikipedia.org/wiki/Ada_Yonath
    "yonath",

    # Nikolay Yegorovich Zhukovsky (Russian: Никола́й Его́рович Жуко́вский, January 17 1847 – March 17, 1921) was a Russian scientist, mathematician and engineer, and a founding father of modern aero- and hydrodynamics. Whereas contemporary scientists scoffed at the idea of human flight, Zhukovsky was the first to undertake the study of airflow. He is often called the Father of Russian Aviation. https://en.wikipedia.org/wiki/Nikolay_Yegorovich_Zhukovsky
    "zhukovsky",
])


def setup_run_dir(runs_dirpath, run_name=None, new_run=False, check_exists=False):
    """
    If new_run is True, creates a new directory:
        If run_name is None, generate a random name
        else build the created directory name with run_name

    If new_run is False, return an existing directory:
        if run_name is None, return the last created directory (from timestamp)
        else return the last created directory (from timestamp) whose name starts with run_name,
        if that does not exist and check_exists is False create a new run with run_name,
        if check_exists is True, then raise an error.

    Special case: if there is no existing runs, the new_run is not taken into account and the function behaves like new_run is True.

    :param runs_dirpath: Parent directory path of all the runs
    :param run_name:
    :param new_run:
    :param check_exists:
    :return: Run directory path. The directory name is in the form "run_name | timestamp"
    """
    # Create runs directory of it does not exist
    if not os.path.exists(runs_dirpath):
        os.makedirs(runs_dirpath, exist_ok=True)

    existing_run_dirnames = os.listdir(runs_dirpath)
    if new_run or (not new_run and not 0 < len(existing_run_dirnames)):
        if run_name is not None:
            # Create another directory name for the run, with its name starting with run_name
            name_timestamped = create_name_timestamped(run_name)
        else:
            # Create another directory name for the run, excluding the existing names
            existing_run_names = [existing_run_dirname.split(" _ ")[0] for existing_run_dirname in
                                  existing_run_dirnames]
            name_timestamped = create_free_name_timestamped(exclude_list=existing_run_names)
        current_run_dirpath = os.path.join(runs_dirpath, name_timestamped)
        os.mkdir(current_run_dirpath)
    else:
        if run_name is not None:
            # Pick run dir based on run_name
            filtered_existing_run_dirnames = [existing_run_dirname for existing_run_dirname in existing_run_dirnames if
                                              existing_run_dirname.split(" _ ")[0] == run_name]
            if filtered_existing_run_dirnames:
                filtered_existing_run_timestamps = [filtered_existing_run_dirname.split(" _ ")[1] for
                                                    filtered_existing_run_dirname in
                                                    filtered_existing_run_dirnames]
                filtered_last_index = filtered_existing_run_timestamps.index(max(filtered_existing_run_timestamps))
                current_run_dirname = filtered_existing_run_dirnames[filtered_last_index]
            else:
                if check_exists:
                    raise FileNotFoundError("Run '{}' does not exist.".format(run_name))
                else:
                    return setup_run_dir(runs_dirpath, run_name=run_name, new_run=True)
        else:
            # Pick last run dir based on timestamp
            existing_run_timestamps = [existing_run_dirname.split(" | ")[1] for existing_run_dirname in
                                       existing_run_dirnames]
            last_index = existing_run_timestamps.index(max(existing_run_timestamps))
            current_run_dirname = existing_run_dirnames[last_index]
        current_run_dirpath = os.path.join(runs_dirpath, current_run_dirname)
    return current_run_dirpath


def create_name_timestamped(name):
    timestamp = time.time()
    formatted_timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    name_timestamped = name + " | " + formatted_timestamp
    return name_timestamped


def create_free_name_timestamped(exclude_list=None):
    if exclude_list is not None:
        names = list(NAME_SET - set(exclude_list))
    else:
        names = list(NAME_SET)
    assert 0 < len(names), \
        "In create_free_name_timestamped(), all possible names have been used. " \
        "Cannot create a new name without a collision! Delete some runs to continue..."
    sorted_names = sorted(names)
    name = sorted_names[0]
    name_timestamped = create_name_timestamped(name)
    return name_timestamped


def setup_run_subdir(run_dir, subdirname):
    subdirpath = os.path.join(run_dir, subdirname)
    os.makedirs(subdirpath, exist_ok=True)
    return subdirpath


def setup_run_subdirs(run_dir, logs_dirname="logs", checkpoints_dirname="checkpoints"):
    logs_dir = os.path.join(run_dir, logs_dirname)
    checkpoints_dir = os.path.join(run_dir, checkpoints_dirname)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    return logs_dir, checkpoints_dir


def wipe_run_subdirs(run_dir, logs_dirname="logs", checkpoints_dirname="checkpoints"):
    logs_dir = os.path.join(run_dir, logs_dirname)
    checkpoints_dir = os.path.join(run_dir, checkpoints_dirname)
    python_utils.wipe_dir(logs_dir)
    python_utils.wipe_dir(checkpoints_dir)


def save_config(config, config_dirpath):
    filepath = os.path.join(config_dirpath, 'config.json')
    with open(filepath, 'w') as outfile:
        json.dump(config, outfile)
    # shutil.copyfile(os.path.join(project_dir, "config.py"), os.path.join(current_logs_dir, "config.py"))


def load_config(config_name="config", config_dirpath="", try_default=False):
    if os.path.splitext(config_name)[1] == ".json":
        config_filepath = os.path.join(config_dirpath, config_name)
    else:
        config_filepath = os.path.join(config_dirpath, config_name + ".json")

    try:
        with open(config_filepath, 'r') as f:
            minified = jsmin(f.read())
            try:
                config = json.loads(minified)
            except json.decoder.JSONDecodeError as e:
                print_utils.print_error("ERROR: Parsing config failed:")
                print(e)
                print_utils.print_info("Minified JSON causing the problem:")
                print(str(minified))
                exit()
        return config
    except FileNotFoundError:
        if config_name == "config" and config_dirpath == "":
            print_utils.print_warning(
                "WARNING: the default config file was not found....")
            return None
        elif try_default:
            print_utils.print_warning(
                "WARNING: config file {} was not found, opening default config file config.defaults.json instead.".format(
                    config_filepath))
            return load_config()
        else:
            print_utils.print_warning(
                "WARNING: config file {} was not found.".format(config_filepath))
            return None


def _merge_dictionaries(dict1, dict2):
    """
    Recursive merge dictionaries.

    :param dict1: Base dictionary to merge.
    :param dict2: Dictionary to merge on top of base dictionary.
    :return: Merged dictionary
    """
    for key, val in dict1.items():
        if isinstance(val, dict):
            dict2_node = dict2.setdefault(key, {})
            _merge_dictionaries(val, dict2_node)
        else:
            if key not in dict2:
                dict2[key] = val

    return dict2


def load_defaults_in_config(config: dict, filepath_key: str="defaults_filepath", depth: int=0) -> dict:
    """
    Searches in the config dict for keys equal to "defaults_filepath".
    When one is found, read the json at the defaults_filepath and add the defaults for the current params.

    Example:
    config = {
        "some_params": {
            filepath_key: "path/to/default_params.json",
            "param_2": {
                "sub_param_1": 0,
                "sub_param_2": 0,
            }
        }
    }
    and there is a file at path/to/default_params.json which reads:
    {
        "param_1": 0,
        "param_2": {
            "sub_param_2": 1,
        }
    }

    The returned config dict will be:
    {
        "some_params": {
            "param_1": 0,
            "param_2": {
                "sub_param_1": 0,
                "sub_param_2": 1,
            }
        }
    }
    Note the value of param_2.sub_param_2 whose default was overwritten while param_2.sub_param_1 was left to default

    @param config:
    @param filepath_key:
    @return: config with all defaults loaded
    """
    assert isinstance(config, dict), f"config should be of type dict, not {type(config)}"

    if filepath_key in config and isinstance(config[filepath_key], str):
        defaults = python_utils.load_json(config[filepath_key])
        if defaults:
            tabs = "\t"*depth
            print_utils.print_info(f"{tabs}INFO: Loading defaults from {config[filepath_key]}")
            # Recursively process defaults
            defaults = load_defaults_in_config(defaults, filepath_key=filepath_key, depth=depth+1)
            # Copy defaults in config if the key is not already there
            config = _merge_dictionaries(defaults, config)
            # for default_name, default_value in defaults.items():
            #     if default_name not in config:
            #         config[default_name] = default_value
            # Delete filepath_key key
            del config[filepath_key]
        else:
            print_utils.print_error(f"ERROR: Could not load defaults from {config[filepath_key]}!")
            raise ValueError

    # Check items of all other keys
    for key, item in config.items():
        if type(item) == dict:
            config[key] = load_defaults_in_config(item, filepath_key=filepath_key, depth=depth+1)

    return config
