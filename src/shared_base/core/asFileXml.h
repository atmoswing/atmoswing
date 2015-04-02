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
 * Portions Copyright 2015 Pascal Horton, Terr@num.
 */
 
#ifndef ASFILEXML_H
#define ASFILEXML_H

#include <wx/xml/xml.h>

#include "asIncludes.h"
#include <asFile.h>


class asFileXml : public asFile
{
public:
    /** Default constructor */
    asFileXml(const wxString &FileName, const ListFileMode &FileMode);

    /** Default destructor */
    virtual ~asFileXml();

    /** Open file */
    virtual bool Open();

    /** Close file */
    virtual bool Close();

    /** Save file */
    bool Save();

    wxXmlNode * GetRoot()
    {
        wxASSERT(m_document.GetRoot());
        return m_document.GetRoot();
    }
    
    void AddChild(wxXmlNode* node);

    bool CheckRootElement();

    wxXmlNode * CreateNodeWithValue(const wxString &name, const bool &content);

    wxXmlNode * CreateNodeWithValue(const wxString &name, const int &content);

    wxXmlNode * CreateNodeWithValue(const wxString &name, const float &content);

    wxXmlNode * CreateNodeWithValue(const wxString &name, const double &content);

    wxXmlNode * CreateNodeWithValue(const wxString &name, const wxString &content);

    bool IsAnAtmoSwingFile();

    bool FileVersionIsOrAbove(const float version);

    void UnknownNode(wxXmlNode *node);

    static bool GetBool(wxXmlNode *node, const bool defaultValue = false);

    static int GetInt(wxXmlNode *node, const int defaultValue = 0);

    static float GetFloat(wxXmlNode *node, const float defaultValue = 0.0f);

    static double GetDouble(wxXmlNode *node, const double defaultValue = 0.0);

    static wxString GetString(wxXmlNode *node, const wxString &defaultValue = wxEmptyString);

protected:
private:
    wxXmlDocument m_document;

};

#endif
