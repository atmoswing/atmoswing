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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 */

#ifndef ASEXCEPTION_H
#define ASEXCEPTION_H

#define asThrowException(msg) \
    throw asException(msg, __FILE__, __LINE__)

#define asThrowExceptionChain(msg, pprevexception) \
    throw asException(msg, __FILE__, __LINE__, pprevexception)

#include <asIncludes.h>

class asException
        : public std::exception
{
public:
    typedef struct//!< Structure for older exceptions
    {
        wxString Message;
        wxString FileName;
        int LineNum;
    } PrevExceptions;

    asException();

    asException(const wxString &message, const char *filename, unsigned int line);

    asException(const std::string &message, const char *filename, unsigned int line);

    asException(const char *message, const char *filename, unsigned int line);

    asException(const wxString &message, const char *filename, unsigned int line, asException prevexception);

    asException(const std::string &message, const char *filename, unsigned int line, asException prevexception);

    asException(const char *message, const char *filename, unsigned int line, asException prevexception);

    virtual ~asException() throw();

    wxString GetMessage() const
    {
        return m_message;
    }

    wxString GetFileName() const
    {
        return m_fileName;
    }

    int GetLineNum() const
    {
        return m_lineNum;
    }

    bool HasChild() const
    {
        return m_previous.size() > 0;
    }

    virtual const char *what() const throw()
    {
        return m_message.char_str();

    }

    wxString GetFullMessage() const;

protected:

private:
    std::vector<PrevExceptions *> m_previous;
    wxString m_message;
    wxString m_fileName;
    int m_lineNum;
};

#endif
