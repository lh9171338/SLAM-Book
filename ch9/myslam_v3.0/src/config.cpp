//
// Created by lihao on 19-9-30.
//

#include "config.h"

namespace myslam{

shared_ptr<Config> Config::Ptr = nullptr;

void Config::setParameterFile(string filename)
{
    if(Ptr == nullptr )
        Ptr = shared_ptr<Config>(new Config);
    Ptr->m_fs = FileStorage(filename.c_str(), FileStorage::READ);
    if(Ptr->m_fs.isOpened() == false)
    {
        cerr << "parameter file " << filename << " does not exist." << endl;
        Ptr->m_fs.release();
        return;
    }
}

Config::~Config()
{
    if(m_fs.isOpened())
        m_fs.release();
}

}