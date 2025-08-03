## Bad numerical computing
- The Patriot Missile Failure:
    (The number 1/10 equals 1/24+1/25+1/28+1/29+1/212 +1/213+.... In other words, the binary expansion of 1/10 is 0.0001100110011001100110011001100.... Now the 24 bit register in the Patriot stored instead 0.00011001100110011001100 introducing an error of 0.0000000000000000000000011001100... binary, or about 0.000000095 decimal. 
    Multiplying by the number of tenths of a second in 100 hours gives 0.000000095×100×60×60×10=0.34.) 
    A Scud travels at about 1,676 meters per second, and so travels more than half a kilometer in this time. 
    This was far enough that the incoming Scud was outside the "range gate" that the Patriot tracked.[^1]

- The Explosion of the Ariane 5
    Specifically a 64 bit floating point number relating to the horizontal velocity of the rocket with respect to the platform was converted to a 16 bit signed integer.
    The number was larger than 32,768, the largest integer storable in a 16 bit signed integer, and thus the conversion failed.[^1]

- The sinking of the Sleipner A offshore platform
    he wall failed as a result of a combination of a serious error in the finite element analysis and insufficient anchorage of the reinforcement in a critical zone. A better idea of what was involved can be obtained from this photo and sketch of the platform. Sleipner Sleipner A The top deck weighs 57,000 tons, and provides accommodation for about 200 people and support for drilling equipment weighing about 40,000 tons. When the first model sank in August 1991, the crash caused a seismic event registering 3.0 on the Richter scale, and left nothing but a pile of debris at 220m of depth. The failure involved a total economic loss of about $700 million.
    
    The post accident investigation traced the error to inaccurate finite element approximation of the linear elastic model of the tricell (using the popular finite element program NASTRAN). The shear stresses were underestimated by 47%, leading to insufficient design. In particular, certain concrete walls were not thick enough. More careful finite element analysis, made after the accident, predicted that failure would occur with this design at a depth of 62m, which matches well with the actual occurrence at 65m.[^1]


[^1] [Some disasters attributable to bad numerical computing](https://www.iro.umontreal.ca/~mignotte/IFT2425/Disasters.html)