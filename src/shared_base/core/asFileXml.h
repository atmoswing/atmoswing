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

#include <wx/xml/xml.h>

#include "asFile.h"

/**
 * @brief XML file class.
 *
 * This class is a wrapper around the wxXmlDocument library.
 */
class asFileXml : public asFile {
  public:
    asFileXml(const wxString& fileName, const FileMode& fileMode);

    ~asFileXml() override = default;

    [[nodiscard]] bool Open() override;

    [[nodiscard]] bool Close() override;

    [[nodiscard]] bool Save();

    [[nodiscard]] wxXmlNode* GetRoot() const {
        wxASSERT(_document.GetRoot());
        return _document.GetRoot();
    }

    void AddChild(wxXmlNode* node);

    [[nodiscard]] virtual bool CheckRootElement() const;

    wxXmlNode* CreateNode(const wxString& name, const bool& content);

    wxXmlNode* CreateNode(const wxString& name, const int& content);

    wxXmlNode* CreateNode(const wxString& name, const float& content);

    wxXmlNode* CreateNode(const wxString& name, const double& content);

    wxXmlNode* CreateNode(const wxString& name, const wxString& content);

    [[nodiscard]] bool IsAnAtmoSwingFile() const;

    [[nodiscard]] bool FileVersionIsOrAbove(float version) const;

    void UnknownNode(wxXmlNode* node);

    [[nodiscard]] static bool GetBool(wxXmlNode* node, bool defaultValue = false);

    [[nodiscard]] static int GetInt(wxXmlNode* node, int defaultValue = 0);

    [[nodiscard]] static float GetFloat(wxXmlNode* node, float defaultValue = 0.0f);

    [[nodiscard]] static double GetDouble(wxXmlNode* node, double defaultValue = 0.0);

    [[nodiscard]] static wxString GetString(wxXmlNode* node, const wxString& defaultValue = wxEmptyString);

    [[nodiscard]] bool GetAttributeBool(wxXmlNode* node, const wxString& attribute, bool defaultValue = false,
                                        bool raiseWarning = true);

    [[nodiscard]] int GetAttributeInt(wxXmlNode* node, const wxString& attribute);

    [[nodiscard]] float GetAttributeFloat(wxXmlNode* node, const wxString& attribute);

    [[nodiscard]] double GetAttributeDouble(wxXmlNode* node, const wxString& attribute);

    [[nodiscard]] wxString GetAttributeString(wxXmlNode* node, const wxString& attribute);

  protected:
  private:
    wxXmlDocument _document;
};

#endif
