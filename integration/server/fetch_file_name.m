function response = fetch_file_name(endpointURL, queryParams)
    % fetch_file_name Sends a GET request to a server endpoint with optional query parameters
    % Inputs:
    %   endpointURL - The URL of the server endpoint (string)
    %   queryParams - A structure containing query parameters (optional)
    % Outputs:
    %   response    - The server's response to the GET request
    
    % Configure options for the GET request
    options = weboptions('Timeout', 15); % Set timeout in seconds
    
    try
        % Send GET request with or without query parameters
        if nargin < 2 || isempty(queryParams)
            % If no query parameters, make the request without them
            response = webread(endpointURL, options);
        else
            % If query parameters are provided, include them in the request
            response = webread(endpointURL, queryParams, options);
        end
        
        fprintf('GET request successful. Server responded with:\n');
        disp(response);
    catch ME
        fprintf('GET request failed.\n');
        disp(ME.message);
        response = [];
    end
end
