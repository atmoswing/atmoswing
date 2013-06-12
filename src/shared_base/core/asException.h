/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef ASEXCEPTION_H
#define ASEXCEPTION_H

#define asThrowException(msg) \
    throw asException(msg, __FILE__, __LINE__)

#define asThrowExceptionChain(msg, pprevexception) \
    throw asException(msg, __FILE__, __LINE__, pprevexception)

#include <asIncludes.h>

class asException : public std::exception
{
public:

    typedef struct//!< Structure for older exceptions
    {
        wxString Message;
        wxString FileName;
        int LineNum;
    } PrevExceptions;

    /** Empty constructor
     * \param message The exception message
     * \param filename The filename where the exception occured
     * \param line The line number where the exception occured
     */
    asException();

    /** Default constructor
     * \param message The exception message
     * \param filename The filename where the exception occured
     * \param line The line number where the exception occured
     */
    asException(const wxString &message, const char *filename, unsigned int line);

    /** Alternative constructor
     * \param message The exception message
     * \param filename The filename where the exception occured
     * \param line The line number where the exception occured
     */
    asException(const std::string &message, const char *filename, unsigned int line);

    /** Alternative constructor
     * \param message The exception message
     * \param filename The filename where the exception occured
     * \param line The line number where the exception occured
     */
    asException(const char *message, const char *filename, unsigned int line);

    /** Default constructor
     * \param message The exception message
     * \param filename The filename where the exception occured
     * \param line The line number where the exception occured
     */
    asException(const wxString &message, const char *filename, unsigned int line, asException prevexception);

    /** Alternative constructor
     * \param message The exception message
     * \param filename The filename where the exception occured
     * \param line The line number where the exception occured
     */
    asException(const std::string &message, const char *filename, unsigned int line, asException prevexception);

    /** Alternative constructor
     * \param message The exception message
     * \param filename The filename where the exception occured
     * \param line The line number where the exception occured
     */
    asException(const char *message, const char *filename, unsigned int line, asException prevexception);

    /** Default destructor
     *  \link http://www.agapow.net/programming/cpp/looser-throw-specifier
     */
    virtual ~asException() throw ();

    /** Access m_Message
     * \return The current value of m_Message
     */
    wxString GetMessage()
    {
        return m_Message;
    }

    /** Set m_Message
     * \param val New value to set
     */
    void SetMessage(const wxString &val)
    {
        m_Message = val;
    }

    /** Access m_FileName
     * \return The current value of m_FileName
     */
    wxString GetFileName()
    {
        return m_FileName;
    }

    /** Access m_LineNum
     * \return The current value of m_LineNum
     */
    int GetLineNum()
    {
        return m_LineNum;
    }

    /** Access m_HasChild
     * \return The current value of m_HasChild
     */
    bool GetHasChild()
    {
        return m_HasChild;
    }


    /** The virtual std what() member */
    virtual const char* what() const throw()
    {
        return m_Message.char_str();

    }

    /** Get the full error message
     * \return A wxString with all information whithin
     */
    wxString GetFullMessage();

protected:

private:
    std::vector < PrevExceptions* > m_Previous; //!< Member variable "m_Previous"
    wxString m_Message; //!< Member variable "m_Message"
    wxString m_FileName; //!< Member variable "m_FileName"
    int m_LineNum; //!< Member variable "m_LineNum"
    bool m_HasChild; //!< Member variable "m_IsReal"
};

#endif
