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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef AS_FILE_XML_H
#define AS_FILE_XML_H

#include <asFile.h>
#include <wx/xml/xml.h>

#include "asIncludes.h"

class asFileXml : public asFile {
   public:
    asFileXml(const wxString &fileName, const FileMode &fileMode);

    ~asFileXml() override = default;

    bool Open() override;

    bool Close() override;

    bool Save();

    wxXmlNode *GetRoot() const {
        wxASSERT(m_document.GetRoot());
        return m_document.GetRoot();
    }

    void AddChild(wxXmlNode *node);

    virtual bool CheckRootElement() const;

    wxXmlNode *CreateNodeWithValue(const wxString &name, const bool &content);

    wxXmlNode *CreateNodeWithValue(const wxString &name, const int &content);

    wxXmlNode *CreateNodeWithValue(const wxString &name, const float &content);

    wxXmlNode *CreateNodeWithValue(const wxString &name, const double &content);

    wxXmlNode *CreateNodeWithValue(const wxString &name, const wxString &content);

    bool IsAnAtmoSwingFile() const;

    bool FileVersionIsOrAbove(float version) const;

    void UnknownNode(wxXmlNode *node);

    static bool GetBool(wxXmlNode *node, bool defaultValue = false);

    static int GetInt(wxXmlNode *node, int defaultValue = 0);

    static float GetFloat(wxXmlNode *node, float defaultValue = 0.0f);

    static double GetDouble(wxXmlNode *node, double defaultValue = 0.0);

    static wxString GetString(wxXmlNode *node, const wxString &defaultValue = wxEmptyString);

    bool GetAttributeBool(wxXmlNode *node, const wxString &attribute, bool defaultValue = false,
                          bool raiseWarning = true);

    int GetAttributeInt(wxXmlNode *node, const wxString &attribute);

    float GetAttributeFloat(wxXmlNode *node, const wxString &attribute);

    double GetAttributeDouble(wxXmlNode *node, const wxString &attribute);

    wxString GetAttributeString(wxXmlNode *node, const wxString &attribute);

   protected:
   private:
    wxXmlDocument m_document;
};

#endif
