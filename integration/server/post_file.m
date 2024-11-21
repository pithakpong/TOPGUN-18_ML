function post_file(filePath, sendURL)
    % Read file data as bytes
    fileBytes = get_file_bytes(filePath);
    
    % Encode file data as base64
    fileContentBase64 = matlab.net.base64encode(fileBytes);
    
    % Create JSON payload
    [~, filename, ext] = fileparts(filePath);
    data.filename = [filename ext];
    data.file_content = fileContentBase64;
    
    % Send POST request with JSON payload
    options = weboptions('MediaType', 'application/json');
    response = webwrite(sendURL, data, options);
    
    % Display server response
    disp(response);
end

function fileBytes = get_file_bytes(filePath)
    % Attempt to open the file
    fid = fopen(filePath, 'rb');
    if fid == -1
        error('Failed to open file: %s. Check if the file exists and the path is correct.', filePath);
    end
    
    % Read file as binary bytes
    fileBytes = fread(fid, '*uint8');
    fclose(fid);
end
