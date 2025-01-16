% Nama file Excel
file_name = 'Data_curah_hujan.xlsx';

if ~isfile(file_name)
    error('File %s tidak ditemukan di direktori kerja.', file_name);
end

% Muat dataset "data_latih"
latih_data = readtable(file_name, 'Sheet', 'data_latih');
disp('Data Latih:');
disp(head(latih_data)); % Tampilkan beberapa baris pertama

% Muat dataset "data_uji"
uji_data = readtable(file_name, 'Sheet', 'data_uji');
disp('Data Uji:');
disp(head(uji_data)); % Tampilkan beberapa baris pertama

% Konversi tabel menjadi matriks jika diperlukan
latih_matrix = table2array(latih_data);
uji_matrix = table2array(uji_data);

% Tampilkan ukuran masing-masing dataset untuk verifikasi
fprintf('Data Latih: %d baris dan %d kolom.\n', size(latih_matrix));
fprintf('Data Uji: %d baris dan %d kolom.\n', size(uji_matrix));

% Pastikan data sudah dimuat dan dikonversi ke matriks sebelumnya
% Pisahkan input dan target dari data latih
train_inputs = latih_matrix(:, 1:12); % 12 input features
train_targets = latih_matrix(:, end); % Target (kolom terakhir)

% Pisahkan input dan target dari data uji
test_inputs = uji_matrix(:, 1:12); % 12 input features
test_targets = uji_matrix(:, end); % Target (kolom terakhir)

% Definisikan arsitektur jaringan saraf
hiddenLayerSize = 10; % Jumlah neuron pada hidden layer

% Buat jaringan saraf
net = feedforwardnet(hiddenLayerSize); % Jaringan dengan satu hidden layer

% Atur fungsi aktivasi untuk setiap layer
net.layers{1}.transferFcn = 'logsig'; % Hidden layer menggunakan log-sigmoid
net.layers{2}.transferFcn = 'purelin'; % Output layer menggunakan fungsi linier

% Atur parameter pelatihan
net.trainParam.epochs = 1000; % Batas maksimal epoch
net.trainParam.goal = 0.001; % Target minimal rata-rata error (MSE)
% Learning rate (lr) dan momentum coefficient (mc) dibuat default
% Default lr: 0.01, Default mc: 0.9

% Latih jaringan menggunakan data latih
[net, tr] = train(net, train_inputs', train_targets');

% Prediksi data uji menggunakan jaringan terlatih
predictions = net(test_inputs')';

% Evaluasi kinerja jaringan
mseError = mse(test_targets - predictions);

% Tampilkan hasil
fprintf('Mean Squared Error (MSE) pada data uji: %.4f\n', mseError);
disp('Perbandingan Target vs Prediksi:');
disp(table(test_targets, predictions, 'VariableNames', {'Actual', 'Predicted'}));

% 1. Visualisasi Sebaran Data (Histogram)
figure;
for i = 1:12
    subplot(4, 3, i); % Buat subplot 4x3
    histogram(latih_matrix(:, i), 20); % Histogram untuk setiap fitur
    title(['Fitur ', num2str(i)]);
    xlabel('Nilai');
    ylabel('Frekuensi');
end
sgtitle('Sebaran Data Latih untuk Setiap Fitur');

% 2. Scatter Plot Input vs Target
figure;
scatter(1:length(train_targets), train_targets, 'b', 'DisplayName', 'Target');
hold on;
scatter(1:length(train_targets), net(train_inputs')', 'r', 'DisplayName', 'Prediksi');
xlabel('Indeks Data');
ylabel('Nilai Target');
legend('Location', 'best');
title('Perbandingan Target dan Prediksi pada Data Latih');

% 3. Plot Mean Squared Error (MSE) Selama Pelatihan
figure;
plot(tr.perf, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Mean Squared Error (MSE)');
title('Performa Pelatihan (MSE)');
grid on;

% 4. Scatter Plot Data Uji (Target vs Prediksi)
figure;
scatter(test_targets, predictions, 'filled');
xlabel('Target Sebenarnya');
ylabel('Prediksi');
title('Scatter Plot Target vs Prediksi pada Data Uji');
grid on;
