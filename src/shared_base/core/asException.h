/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
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

    /** Access m_message
     * \return The current value of m_message
     */
    wxString GetMessage()
    {
        return m_message;
    }

    /** Set m_message
     * \param val New value to set
     */
    void SetMessage(const wxString &val)
    {
        m_message = val;
    }

    /** Access m_fileName
     * \return The current value of m_fileName
     */
    wxString GetFileName()
    {
        return m_fileName;
    }

    /** Access m_lineNum
     * \return The current value of m_lineNum
     */
    int GetLineNum()
    {
        return m_lineNum;
    }

    /** Access m_hasChild
     * \return The current value of m_hasChild
     */
    bool GetHasChild()
    {
        return m_hasChild;
    }


    /** The virtual std what() member */
    virtual const char* what() const throw()
    {
        return m_message.char_str();

    }

    /** Get the full error message
     * \return A wxString with all information whithin
     */
    wxString GetFullMessage();

protected:

private:
    std::vector < PrevExceptions* > m_previous; //!< Member variable "m_previous"
    wxString m_message; //!< Member variable "m_message"
    wxString m_fileName; //!< Member variable "m_fileName"
    int m_lineNum; //!< Member variable "m_lineNum"
    bool m_hasChild; //!< Member variable "m_isReal"
};

#endif
