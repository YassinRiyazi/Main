## Introduction 
Instructor: Andreas Best

## Caution
- Keep the light source running for at least 30 minutes, because shorter periods can significantly reduce the life expectancy of the special xenon lamp.
- The program only runs if the dongle, which is used as software protection, is plugged in.
- The software μSurf® serves both for the control of the measurement process and the basic evaluation of measurement results. As delivered, the software is prepared for use. For security, a CD with the entire software is included.

> [!IMPORTANT]
> ## Doing Experiments
> 
>    ### Turning System ON
>    - Turn on main switch in the cabinet
>    - Turn on the light
>    - Enter USER and Pass
>    - Open μsoft application
>        - If the master control has just been switched on, "Set reference" request should always be answered with “Yes”.
>        - The reference position is the home position of the sample stage (x- and y-axis) and the linear axis of the sensor (z-axis). The coordinates of the home position are x = y = z = 0. Move it to most bottom and left.
>    >[!CAUTION]
>    - By setting the reference position a software stop is initialized, because the software doesn’t accept negative coordinates. In case of an accurate setting of the reference position, an accidental collision of the objective and the x/y-stage can be prevented.
>     
> 
>    ### Doing a measurement
> 
> 
> 
> 
>    ### Reference Calibration
>     In standard microscope mode (see ‘Doing a measurement’, p. 17) the focus plane, i.e. a sharp picture of the sample surface, should be found by moving the measuring head up and down by means of the joystick.
>     Then switch over to confocal mode, i.e. click the appropriate button in the software. 
>     The Nipkow disk is now situated in the light path. 
>     Look for the brightest area while moving through the confocal levels and adjust the lighting.
>     Then search for the upper and lower limits of the measurement range and enter these data.
>     As desired, enter some further measurement parameters and start the measurement.
> 
> 
> 

## General Information 
-   The base of the system is a solid granite stone slab for damping oscillations.
-   The required white light is generated in an external xenon light source, and transmitted by a fibre light guide to the measuring head.

    | Microscope objective*                             | 800-S / 800-L (20X)   | 320-S / 320-L (50X)   | 160-S / 160-L (100X)  |
    |---------------------------------------------------|-----------------------|-----------------------|-----------------------|
    | Measurement field [µm] × [µm]                     | 800 × 772             | 320 × 308.8           | 160 × 154.4           |
    | Maximum area in stitching mode [mm²]              | 46.8                  | 7.3                   | 1.8                   |
    | Working distance [mm]                             | 3.1 / 12.0            | 0.66 / 10.6           | 0.31 / 3.4            |
    | Numeric aperture                                  | 0.46 / 0.40           | 0.80 / 0.50           | 0.95 / 0.80           |
    | Maximum surface slope for specular reflection [°] | 13.7 / 11.7           | 26.6 / 15.0           | 35.9 / 26.6           |
    | Maximum vertical measurement range [mm]           | 0.33 / 0.5            | 0.11 / 0.25           | 0.08 / 0.11           |     
    | Vertical resolution** [nm]                        | 5 / 6                 | 2 / 4                 | 1.5 / 2               |

    \* xxx-S = standard type, xxx-L = long working distance  
    \** piezo driven measurement

-   A scan requires 3 Bytes memory space per measurement point (i.e. 1,5 kBytes for a profile with 512 values and 768 kBbytes for a measurement.). In case of stitched measurements, the memory requirement increases by a factor of the number of measured images (e.g. 2 × 2 stitching; 4 × 768 kbytes = 3MBytes)

    ### Device Limitation
    -   which is moved in the z-direction by a linear step drive positioner (positioning range: 100 mm).
    -   Additionally a piezo drive, which only moves the objective, can move over a stroke of 350 μm (standard version).
    -   The software now controls the further course of measurement. A number of up to 1000 pictures are recorded by the frame grabber and stored in the computer for further processing.
    -   
    -   
    ### [Nipkow disk](https://en.wikipedia.org/wiki/Nipkow_disk)
        A Nipkow disk (sometimes Anglicized as Nipkov disk; patented in 1884), also known as scanning disk, is a mechanical, rotating, geometrically operating image scanning device, patented by Paul Gottlieb Nipkow in Berlin.[1] 
        This scanning disk was a fundamental component in mechanical television, and thus the first televisions, through the 1920s and 1930s.
        WIKIPEDIA

        Apart from the aforementioned mechanical television, which did not become popular for the practical reasons mentioned above, a Nipkow disk is used in one type of confocal microscope, a powerful optical microscope.
