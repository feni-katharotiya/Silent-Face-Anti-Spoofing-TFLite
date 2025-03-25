================== TO GENERATE IV & KEY ===========================
openssl enc -aes-128-cbc -k secret -P -md sha1 -pbkdf2 -iter 10000
================== TO GENERATE IV & KEY ===========================

================== For Encrypting Models(.pb/.tflite) ===========================
1. Run ModelEncryption.exe
2. Do you want to encrypt for open vino ? Ans - n
3. Give the path for the model to encrypt
4. Give the path for the model to save (With model name)
5. Provide the iv key specified in the key file.
6. provide the key specified in the key file.
7. Encrypted model can be found at the save location.
================== For Encrypting Models(.pb/.tflite) ===========================

================== For Decrypting Models(.pb/.tflite) ===========================
1. Run ModelEncryption.exe
2. Do you want to decrypt for open vino ? Ans - n
3. Give the path for the model for encrypted model
4. Give the path for the model to save (With model name)
5. Provide the iv key specified in the key file.
6. provide the key specified in the key file.
7. Decrypted model can be found at the save location.
================== For Decrypting Models(.pb/.tflite) ===========================

================== For Encrypting Models(OpenVino Model) ===========================
1. Run ModelEncryption.exe
2. Do you want to encrypt for open vino ? Ans - y
3. Give the path for the bin file.
4. Give the path for xml file.
5. Give the path for the model to save (With model name) Note: (Combine bin and xml file will be same as one file)
6. Provide the iv key specified in the key file.
7. provide the key specified in the key file.
8. Encrypted model can be found at the save location.
================== For Encrypting Models(OpenVino Model) ===========================

================== For Decrypting Models(OpenVino Model) ===========================
1. Run ModelEncryption.exe
2. Do you want to encrypt for open vino ? Ans - y
3. Give the path for the model for encrypted model
5. Give the path for the decrypted model to save (With model name without extension) Note: (bin and xml file will be generated)
6. Provide the iv key specified in the key file.
7. provide the key specified in the key file.
8. Decrypted xml and bin file can be found at the save location.
================== For Decrypting Models(OpenVino Model) ===========================