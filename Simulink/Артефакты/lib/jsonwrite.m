function jsonwrite(path, s)
% Записать struct s в файл path как форматированный JSON.
fid = fopen(path, 'w');
fwrite(fid, jsonencode(s, 'PrettyPrint', true));
fclose(fid);
end
