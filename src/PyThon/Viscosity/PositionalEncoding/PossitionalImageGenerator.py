import  cv2
import  numpy               as  np
import  matplotlib.pyplot   as  plt
from    PIL                 import Image

def sinusoidal_positional_encoding(max_position, d_model):
    """
    Task:       
        This is a part of viscosity estimation project.
    Sub-Task:  
        Implement positional encoding for the Markov chain model.

    Description:
        Probably I am going to use CNN and Transformer for this project. I will treat each frame of the video as
        a Markov state and the video as a Markov chain.
        But I need to provide position of the object and its velocity as input features to the model.
        I want to use positional encoding to encode the position and velocity of the object in the video.
    
        Before I have used padding to keep the scale of objects in frames.
        I was thinking about by adding polar/arc like positional encoding maybe I could keep the scale of objects in frames.
        But I am not sure if it is a good idea.
        I will try to implement it and see how it works. Worst case I will use padding again.

    Why positional encoding?
        In the context of transformers, every word in a sentence is mapped to a vector through embeddings. The transformer then uses these vectors to generate keys and queries for its self-attention mechanism.
        The effectiveness of this process hinges on how well the positional encoding adapts to shifts in position., which do not inherently understand the order of input data.
        In a Markov chain model, especially when dealing with time series or sequential data, positional encoding helps the model to:
        - Understand the temporal relationships between states.
        - Capture the dynamics of the system over time.
        - Maintain the order of states in the Markov chain, which is essential for accurate predictions
        - Facilitate the model's ability to generalize from seen to unseen positions in the sequence
        
    Steps:
        1. Deciding on the positional encoding scheme.
            one of the most important concepts of a transformer model — positional encoding. 
            The authors of “Attention is all you need” chose a combination of sinusoidal curves for the positional encoding vectors.

            Encoding must be able to identified the position of a data in a timeseries uniquely. 
            It should be able to expand to arbitrary number of data in a time series.
            It should be compact and efficient. [Not one-hot encoding]
            Binary encoding is not good because it is not smooth and does not generalize well.
                Meaningful encoding: 
                    we want small changes in position to correspond to small changes in representation.
                    The binary method, however, leads to large jumps in encoding for adjacent positions, which can confuse the model as it tries to interpret these as meaningful patterns.

                Continuity:
                    Continuity in positional encoding helps the model to generalize better to positions it hasn’t seen. A continuous encoding scheme means that positions that are numerically close will receive similar representations, and this would aid the model in understanding patterns that span across multiple positions. Lack of smooth transitions also means that the model’s ability to interpolate and extract useful positional information is limited
            
            Encoding scheme should not blow up the size of the input data. So a periodic function is a good choice.
            To extend the encoding to arbitrary number of data in a time series, we can add a perpendicular sine wave with different frequency component to the encoding.
            The first sine wave provides a base positional signal, and by adding a second sine wave at a different frequency, we allow the encoding to oscillate differently as we move along the sequence.

            This additional dimension effectively creates a more complex positional landscape.
            Each position is now represented by a point in this two-dimensional space, where each dimension corresponds to a sine wave of a different frequency.

            * My initial thought was to use the row as the position but I think its better to include time or a dimensionless number of time as the position.

            For a small positional shift δx, the change in positional encoding P should be a linear function.
            Such a transformation ensures that the relationship captured between different parts of the sentence remains consistent, even as inputs vary slightly.
            This consistency is vital for the transformer to maintain syntactic relationships within a sentence across different positions.
            

            x: Length/ Embedding dimension
            y: Speed / Position
            Finding maximum and minimum values of the frame lengths and speeds.

        2. Implement positional encoding for the Markov chain model.
        3. Test the positional encoding with a simple model.
        
                
    Analogies to LNP:
        A sentence:                 A frame of video
        A Token embedding vector:   A row of pixels in a frame
    
    References:
        https://medium.com/@gunjassingh/positional-encoding-in-transformers-a-visual-and-intuitive-guide-0761e655cea7
        https://medium.com/data-science/master-positional-encoding-part-i-63c05d90a0c3
    """

    position = np.arange(max_position)[:, np.newaxis]
    # The original formula pos / 10000^(2i/d_model) is equivalent to pos * (1 / 10000^(2i/d_model)).
    # I use the below version for numerical stability
    div_term = np.exp(np.arange(0, d_model, 2) * - (np.log(100000.0) / d_model))
    
    pe = np.zeros((max_position, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

def cropped_position_encoding(  x1,x2,
                                max_position = 1245,  # Maximum sequence length
                                d_model = 530  ,      # Embedding dimension:
                            ):
    """
    Assumptions:
        Images are unified in size ex. (130, 1248)
    
    Adding position encoding to the images

    For erode results
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/Viscosity/PositionalEncoding/doc/WithErode.png" alt="Italian Trulli">

    And without erode results see the images below.
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/Viscosity/PositionalEncoding/doc/WithoutErode.png" alt="Italian Trulli">

    """
    
    pe = sinusoidal_positional_encoding(max_position, d_model).T
    pe = cv2.resize(pe[:, x1:x2], (x2-x1-1, 130), interpolation=cv2.INTER_LINEAR)
    pe_norm = cv2.normalize(pe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a red-black RGB image
    red_img = np.zeros((pe_norm.shape[0], pe_norm.shape[1], 3), dtype=np.uint8)
    red_img[..., 0] = pe_norm  # Red channel
    red_img[..., 1] = pe_norm  # green channel

    result_img = Image.fromarray(red_img)
    result_img.save('Projects/Viscosity/Markov/fsgfg.png')
    return pe

def main(max_position = 1245,  # Maximum sequence length
         d_model = 530,        # Embedding dimension
         _plot:bool = False):
    """
    Main function to generate sinusoidal positional encoding.
    """

    # Generate positional encoding
    pe = sinusoidal_positional_encoding(max_position, d_model).T
    pe = cv2.normalize(pe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if _plot:
        # Visualize the positional encoding
        plt.figure(figsize=(10, 5))
        plt.imshow(pe, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Sinusoidal Positional Encoding')
        plt.xlabel('Position')
        plt.ylabel('Embedding Dimension')
        plt.show()

    return pe

if __name__ == "__main__":
    pe = main(max_position = 31245, d_model = 530, _plot = False)
    result_img = Image.fromarray(pe.astype(np.uint8))
    result_img.save('Projects/Viscosity/Markov/PositionalFullGray.png')
    print("Positional encoding image saved.")