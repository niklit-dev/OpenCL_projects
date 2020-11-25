#ifndef CLIENT_SOCK_HPP
#define CLIENT_SOCK_HPP

#ifdef _WIN32
	#include <winsock2.h>
	#include <windows.h>
#else
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <unistd.h>
#endif

#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <nlohmann/json.hpp>

#define PAGE 256

using json = nlohmann::json;

class Socket_exch_JSON
{
private:
    char buff[1024];
#ifdef _WIN32
    SOCKET my_sock;
    HOSTENT *hst;
#else
    int my_sock;
#endif
    sockaddr_in dest_addr;
    const char ok[3] = "OK";
    char res_ok[3] = {0};

public:
    Socket_exch_JSON(int port, char *serv_addr)
    {
#ifdef _WIN32
        // Шаг 1 - инициализация библиотеки Winsock
        if (WSAStartup(0x202,(WSADATA *)&buff[0]))
        {
            std::cerr << "WSAStart error " << WSAGetLastError() << std::endl;
            throw std::exception();
        }
#endif
        // Шаг 2 - создание сокета
        my_sock=socket(AF_INET,SOCK_STREAM,0);
        if (my_sock < 0)
        {
#ifdef _WIN32
            std::cerr << "Socket() error " << WSAGetLastError() << std::endl;
#endif
            std::cerr << "Socket() error " << std::endl;
            throw std::exception();
        }
        // Шаг 3 - установка соединения

        // заполнение структуры sockaddr_in
        // указание адреса и порта сервера
        dest_addr.sin_family=AF_INET;
        dest_addr.sin_port=htons(port);
#ifdef _WIN32
        // преобразование IP адреса из символьного в
        // сетевой формат
        if (inet_addr(serv_addr)!=INADDR_NONE)
            dest_addr.sin_addr.s_addr=inet_addr(serv_addr);
        else
            // попытка получить IP адрес по доменному
            // имени сервера
            if (hst=gethostbyname(serv_addr))
                // hst->h_addr_list содержит не массив адресов,
                // а массив указателей на адреса
                ((unsigned long *)&dest_addr.sin_addr)[0]=((unsigned long **)hst->h_addr_list)[0][0];
            else
            {
                closesocket(my_sock);
                WSACleanup();
                throw std::exception();
            }
#else
        dest_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#endif
        // адрес сервера получен – пытаемся установить
        // соединение
        int count = 0;
        while(1){
            if (connect(my_sock,(sockaddr *)&dest_addr, sizeof(dest_addr)) != 0)
            {
                std::cout << "Connect error: " << " Port: " << port << std::endl;
                count++;
                if(count > 10)
                {
                    std::cerr << "Connection Error!" << std::endl;
                    throw std::exception();
                }
            }
            else
            {
                break;
            }
        }
    }

    ~Socket_exch_JSON()
    {
#ifdef _WIN32
        closesocket(my_sock);
        WSACleanup();
#else
        close(my_sock);
#endif
    }

    void close_socket()
    {
#ifdef _WIN32
        closesocket(my_sock);
        WSACleanup();
#else
        close(my_sock);
#endif
    }
	
    // Принимаем заголовок принимаемого пакета
    int recive_head()
    {
        // Принимаем длину пакета в байтах
        char data_len_b[4];
        recv(my_sock, data_len_b, sizeof(data_len_b), 0);
        send(my_sock, ok, 2, 0);
        // Длина пакета данных и int
        int data_len;
        memcpy(&data_len, data_len_b, 4);
        return data_len;
    }

    // Принимаем данные пакета
    void recive_data(std::string &s, const int data_len)
    {
        // Принимаем данные
        char* data_mas = new char[data_len];
        int len_page = data_len;
        // Цикл по размеру страницы
        for(int i=0; i<data_len; i+=PAGE)
        {
            int length=0;
            if(PAGE>len_page)
                length = len_page;
            else
                length = PAGE;
            recv(my_sock, data_mas+i, length, 0);
            len_page = len_page-PAGE;
        }
        send(my_sock, ok, 2, 0);
        // Приводим данные к нужному формату
        s = std::string(data_mas, data_len);

        delete[] data_mas;
    }

    // Отправляем заголовок принимаемого пакета
    void send_head(int data_len)
    {
        // Отправляем длину массива данных в байтах
        char len_out[4];
        res_ok[3] = {0};
        memcpy(len_out, &data_len, 4);
        send(my_sock, len_out, 4, 0);
        recv(my_sock, res_ok, sizeof(res_ok), 0);
        if(strncmp(res_ok, ok, 2) != 0)
        {
            std::cerr << res_ok << std::endl;
            throw std::exception();
        }
    }

    // Отправляем данные одного пакета
    void send_data(char *mas, int data_len)
    {
        res_ok[3] = {0};
        // Отправляем данные (Возможно потребуется цикл по длине страницы)
        send(my_sock, mas, data_len, 0);
        recv(my_sock, res_ok, sizeof(res_ok), 0);
        if(strncmp(res_ok, ok, 2) != 0)
        {
            std::cerr << res_ok << std::endl;
            throw std::exception();
        }
    }

    json recive_json()
    {
        int len_byte = recive_head();
        std::string s;
        recive_data(s, len_byte);
        auto j = json::parse(s);

        return j;
    }

    void send_json(json j)
    {
        std::string s = j.dump();

        send_head(s.length());

        int l = s.length();

        char *c = new char[s.length()+1];
        strcpy(c, s.c_str());

        send_data(c, l);

        delete[] c;
    }
};

#endif
