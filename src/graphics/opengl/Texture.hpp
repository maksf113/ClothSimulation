#pragma once
#include <GL/glew.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "glError.hpp"

class Texture
{
public:
    Texture(size_t width, size_t height, uint32_t internalFormat, uint32_t format, uint32_t type);
    Texture(const std::string& file);
    void bind() const;
    void bind(uint32_t i) const;
    void unbind() const;
    void data(const void* data);
    void wrapping(uint32_t wrapping);
    void minFilter(uint32_t minFilter);
    void magFilter(uint32_t magFilter);
    size_t id() const;
    ~Texture();
private:
    void load(const std::string& file);
private:
    uint32_t m_id;
    size_t m_width;
    size_t m_height;
    uint32_t m_internalFormat;
    uint32_t m_format;
    uint32_t m_type;
};

inline Texture::Texture(size_t width, size_t height, uint32_t internalFormat, uint32_t format, uint32_t type)
    : m_width(width), m_height(height), m_format(format), m_internalFormat(internalFormat), m_type(type)
{
    GL(glGenTextures(1, &m_id));
    bind();
    minFilter(GL_LINEAR_MIPMAP_LINEAR);
    magFilter(GL_LINEAR);
    wrapping(GL_CLAMP_TO_EDGE);
    GL(glGenerateMipmap(GL_TEXTURE_2D));
}
inline Texture::Texture(const std::string& file)
{
    GL(glGenTextures(1, &m_id));
    bind();
    minFilter(GL_LINEAR_MIPMAP_LINEAR);
    magFilter(GL_LINEAR);
    wrapping(GL_REPEAT);
    load(file);
    GL(glGenerateMipmap(GL_TEXTURE_2D));

}
void Texture::bind() const
{
    GL(glBindTexture(GL_TEXTURE_2D, m_id));
}
void Texture::bind(uint32_t i) const
{
    GL(glActiveTexture(GL_TEXTURE0 + i));
    GL(glBindTexture(GL_TEXTURE_2D, m_id));
}
void Texture::unbind() const
{
    GL(glBindTexture(GL_TEXTURE_2D, 0));
}
void Texture::data(const void* data)
{
    GL(glTexImage2D(GL_TEXTURE_2D, 0, m_internalFormat, m_width, m_height, 0, m_format, m_type, data));
}
void Texture::wrapping(uint32_t wrapping)
{
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapping));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapping));
}
inline void Texture::minFilter(uint32_t minFilter)
{
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter));
}
inline void Texture::magFilter(uint32_t magFilter)
{
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, magFilter));
}
size_t Texture::id() const
{
    return m_id;
}
inline Texture::~Texture()
{
    GL(glDeleteTextures(1, &m_id));
}

inline void Texture::load(const std::string& file)
{
    stbi_set_flip_vertically_on_load(true);
    int width, height, chanels;
    unsigned char* textureData = stbi_load(file.c_str(), &width, &height, &chanels, 0);
    if (!textureData)
    {
        printf("%s error: could not load file '%s'\n", __FUNCTION__, file.c_str());
        stbi_image_free(textureData);
    }
    switch (chanels)
    {
    case 1:
        m_format = GL_RED;
        m_internalFormat = GL_R8;
        break;
    case 3:
        m_format = GL_RGB;
        m_internalFormat = GL_RGB8;
        break;
    case 4:
        m_format = GL_RGBA;
        m_internalFormat = GL_RGBA8;
        break;
    }
    if (chanels == 0)
    {
        printf("%s error - unsupported texture format\n", __FUNCTION__);
    }
    m_type = GL_UNSIGNED_BYTE;
    m_width = width;
    m_height = height;
    data(textureData);
    stbi_image_free(textureData);
}