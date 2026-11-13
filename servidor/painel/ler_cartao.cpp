#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <unistd.h>
#include "eidlib.h"
#include "eidlibException.h"

// Função para codificar base64
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const unsigned char* bytes_to_encode, size_t in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; j < i + 1; j++)
            ret += base64_chars[char_array_4[j]];

        while (i++ < 3)
            ret += '=';
    }
    return ret;
}

std::string escape_json(const std::string &s) {
    std::string output;
    for (char c : s) {
        switch (c) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default: output += c; break;
        }
    }
    return output;
}

int calcularIdade(const std::string& birthDate) {
    if (birthDate.length() < 8) return 0;
    int dia = std::stoi(birthDate.substr(0, 2));
    int mes = std::stoi(birthDate.substr(3, 2));
    int ano = std::stoi(birthDate.substr(6, 4));

    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);
    int idade = now->tm_year + 1900 - ano;
    if (now->tm_mon + 1 < mes || (now->tm_mon + 1 == mes && now->tm_mday < dia)) {
        idade--;
    }
    return idade;
}

int main() {
    eIDMW::PTEID_InitSDK();

    try {
        eIDMW::PTEID_ReaderSet &rs = eIDMW::PTEID_ReaderSet::instance();

        // Cada invocação arranca o SDK de raiz (sem estado entre pedidos), e a
        // enumeração de leitores do pcscd por vezes ainda não está pronta no
        // instante do primeiro pedido, lançando EIDMW_ERR_NO_READER mesmo com
        // o leitor fisicamente presente. Repetir por até ~1,5s resolve a
        // corrida sem mascarar uma ausência real de leitor.
        eIDMW::PTEID_ReaderContext *readerPtr = nullptr;
        for (int tentativa = 0; tentativa < 10 && !readerPtr; tentativa++) {
            try {
                readerPtr = &rs.getReader();
            } catch (eIDMW::PTEID_ExNoReader &) {
                usleep(150000);
            }
        }
        if (!readerPtr) throw eIDMW::PTEID_ExNoReader();
        eIDMW::PTEID_ReaderContext &reader = *readerPtr;
        eIDMW::PTEID_EIDCard &card = reader.getEIDCard();
        eIDMW::PTEID_EId &id = card.getID();

        std::string healthNumber = id.getHealthNumber();
        std::string givenName = id.getGivenName();
        std::string surname = id.getSurname();
        std::string name = givenName + " " + surname;
        std::string sex = id.getGender();
        std::string birthDate = id.getDateOfBirth(); // Tem formato dd mm aaaa

        int idade = calcularIdade(birthDate);

        eIDMW::PTEID_Photo& photoObj = id.getPhotoObj();
        eIDMW::PTEID_ByteArray& pngData = photoObj.getphoto(); // PNG
        std::string foto_base64 = base64_encode(pngData.GetBytes(), pngData.Size());

        std::cout << "{";
        std::cout << "\"paciente_id\":\"" << escape_json(healthNumber) << "\",";
        std::cout << "\"nome\":\"" << escape_json(name) << "\",";
        std::cout << "\"sexo\":\"" << escape_json(sex) << "\",";
        std::cout << "\"idade\":" << idade;
        std::cout << ",\"foto\":\"" << foto_base64 << "\"";
        std::cout << "}" << std::endl;
    } catch (eIDMW::PTEID_Exception &e) {
        std::cerr << "PTEID_Exception: code=" << e.GetError()
                   << " msg=" << e.GetMessage() << std::endl;
        std::cout << "{\"paciente_id\":\"\"}" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "std::exception: " << e.what() << std::endl;
        std::cout << "{\"paciente_id\":\"\"}" << std::endl;
    } catch (...) {
        std::cerr << "Excecao desconhecida" << std::endl;
        std::cout << "{\"paciente_id\":\"\"}" << std::endl;
    }

    eIDMW::PTEID_ReleaseSDK();
    return 0;
}
