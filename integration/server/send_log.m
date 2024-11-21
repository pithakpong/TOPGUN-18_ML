function send_log(timestamp, event, sendURL)
    % Create JSON payload
    data.timestamp = timestamp;
    data.event = event;
    
    % Send POST request with JSON payload
    options = weboptions('MediaType', 'application/json');
    response = webwrite(sendURL, data, options);
    
    % Display server response
    disp(response);
end
