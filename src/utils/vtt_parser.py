def extract_text_from_vtt(vtt_file_path, output_txt_file_path=None):
    """
    Extracts only the text content from a WebVTT file, removing timecodes, indices, and blank lines.

    Parameters:
        vtt_file_path (str): The path to the input .vtt file.
        output_txt_file_path (str, optional): The path to save the extracted text file.
                                              If not provided, the function will only return the text.

    Returns:
        str: The extracted text as a string.
        If output_txt_file_path is provided, it also saves the text to a file.
    """
    with open(vtt_file_path, 'r', encoding='utf-8') as vtt_file:
        lines = vtt_file.readlines()
    
    # Filter out lines that are not text
    text_lines = []
    for line in lines:
        if "-->" not in line and not line.strip().isdigit() and line.strip() != "WEBVTT":
            if line.strip():  # Exclude empty lines
                text_lines.append(line.strip())
    
    # Join all text lines into a single string
    extracted_text = "\n".join(text_lines)
    
    # Save the text to a file if output path is provided
    if output_txt_file_path:
        with open(output_txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(extracted_text)
    
    return extracted_text

# Example usage:
# extracted_text = extract_text_from_vtt("input.vtt", "output.txt")
# print(extracted_text)