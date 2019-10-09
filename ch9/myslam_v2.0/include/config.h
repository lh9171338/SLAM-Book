//
// Created by lihao on 19-9-30.
//

#ifndef CONFIG_H
#define CONFIG_H

#include "include.h"

namespace myslam{

class Config
{
private:
    static shared_ptr<Config> Ptr;
    FileStorage m_fs;

public:
    Config() {} // private constructor makes a singleton
    ~Config();  // close the file when deconstructing
    // set a new config file
    static void setParameterFile(string filename);
    // access the parameter values
    template<typename T>
    static T get(string key)
    {
        return T(Ptr->m_fs[key]);
    }
};
}

#endif //CONFIG_H
