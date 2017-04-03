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

#include "asFileParameters.h"

asFileParameters::asFileParameters(const wxString &FileName, const ListFileMode &FileMode)
        : asFileXml(FileName, FileMode)
{
    // FindAndOpen() processed by asFileXml
}

asFileParameters::~asFileParameters()
{
    //dtor
}

VectorInt asFileParameters::BuildVectorInt(int min, int max, int step)
{
    if (min>max) {
        asThrowException(wxString::Format(_("Error when building a vector from the parameters file: min=%d > max=%d."), min, max));
    }
    if (step==0) {
        asThrowException(_("Error when building a vector from the parameters file: step=0."));
    }

    unsigned int stepsnb = (unsigned int) 1 + (max - min) / step;
    VectorInt vect(stepsnb);
    for (int i = 0; i < stepsnb; i++) {
        vect[i] = min + i * step;
    }

    return vect;
}

VectorInt asFileParameters::BuildVectorInt(wxString str)
{
    VectorInt vect;
    wxChar separator = ',';
    while (str.Find(separator) != wxNOT_FOUND) {
        wxString strbefore = str.BeforeFirst(separator);
        str = str.AfterFirst(separator);
        double val;
        strbefore.ToDouble(&val);
        int valint = (int) val;
        vect.push_back(valint);
    }
    if (!str.IsEmpty()) {
        double val;
        str.ToDouble(&val);
        int valint = (int) val;
        vect.push_back(valint);
    }

    return vect;
}

VectorFloat asFileParameters::BuildVectorFloat(float min, float max, float step)
{
    if (min>max) {
        asThrowException(wxString::Format(_("Error when building a vector from the parameters file: min=%.2f > max=%.2f."), min, max));
    }
    if (step==0) {
        asThrowException(_("Error when building a vector from the parameters file: step=0."));
    }

    int stepsnb = (int) (1 + (max - min) / step);
    VectorFloat vect((unsigned long) stepsnb);
    for (int i = 0; i < stepsnb; i++) {
        vect[i] = min + (float) i * step;
    }

    return vect;
}

VectorFloat asFileParameters::BuildVectorFloat(wxString str)
{
    VectorFloat vect;
    wxChar separator = ',';
    while (str.Find(separator) != wxNOT_FOUND) {
        wxString strbefore = str.BeforeFirst(separator);
        str = str.AfterFirst(separator);
        double val;
        strbefore.ToDouble(&val);
        float valfloat = (float) val;
        vect.push_back(valfloat);
    }
    if (!str.IsEmpty()) {
        double val;
        str.ToDouble(&val);
        float valfloat = (float) val;
        vect.push_back(valfloat);
    }

    return vect;
}

VectorDouble asFileParameters::BuildVectorDouble(double min, double max, double step)
{
    if (min>max) {
        asThrowException(wxString::Format(_("Error when building a vector from the parameters file: min=%.2f > max=%.2f."), min, max));
    }
    if (step==0) {
        asThrowException(_("Error when building a vector from the parameters file: step=0."));
    }

    int stepsnb = (int) (1 + (max - min) / step);
    VectorDouble vect(stepsnb);
    for (int i = 0; i < stepsnb; i++) {
        vect[i] = min + (double) i * step;
    }

    return vect;
}

VectorDouble asFileParameters::BuildVectorDouble(wxString str)
{
    VectorDouble vect;
    wxChar separator = ',';
    while (str.Find(separator) != wxNOT_FOUND) {
        wxString strbefore = str.BeforeFirst(separator);
        str = str.AfterFirst(separator);
        double val;
        strbefore.ToDouble(&val);
        vect.push_back(val);
    }
    if (!str.IsEmpty()) {
        double val;
        str.ToDouble(&val);
        vect.push_back(val);
    }

    return vect;
}

VectorString asFileParameters::BuildVectorString(wxString str)
{
    VectorString vect;
    wxChar separator = ',';
    while (str.Find(separator) != wxNOT_FOUND) {
        wxString strBefore = str.BeforeFirst(separator).Trim().Trim(false);
        str = str.AfterFirst(separator).Trim().Trim(false);
        vect.push_back(strBefore);
    }
    if (!str.IsEmpty()) {
        vect.push_back(str);
    }

    return vect;
}

VectorInt asFileParameters::GetVectorInt(wxXmlNode *node)
{
    VectorInt vect;
    wxString nodeName = node->GetName();
    wxString method = node->GetAttribute("method");
    if (method.IsEmpty()) {
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorInt(valueStr);
    } else if (method.IsSameAs("fixed")) {
        long value;
        wxString valueStr = node->GetChildren()->GetContent();
        if (!valueStr.ToLong(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        vect.push_back(int(value));
    } else if (method.IsSameAs("array")) {
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorInt(valueStr);
    } else if (method.IsSameAs("minmax")) {
        long value;

        wxString valueMinStr = node->GetAttribute("min");
        if (!valueMinStr.ToLong(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        int min = (int) value;

        wxString valueMaxStr = node->GetAttribute("max");
        if (!valueMaxStr.ToLong(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        int max = (int) value;

        wxString valueStepStr = node->GetAttribute("step", "1");
        if (!valueStepStr.ToLong(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        int step = (int) value;

        vect = BuildVectorInt(min, max, step);
    } else {
        wxLogVerbose(_("The method is not correctly defined for %s in the calibration parameters file."), nodeName);
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorInt(valueStr);
    }

    return vect;
}

VectorFloat asFileParameters::GetVectorFloat(wxXmlNode *node)
{
    VectorFloat vect;
    wxString nodeName = node->GetName();
    wxString method = node->GetAttribute("method");
    if (method.IsEmpty()) {
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorFloat(valueStr);
    } else if (method.IsSameAs("fixed")) {
        double value;
        wxString valueStr = node->GetChildren()->GetContent();
        if (!valueStr.ToDouble(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        vect.push_back(float(value));
    } else if (method.IsSameAs("array")) {
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorFloat(valueStr);
    } else if (method.IsSameAs("minmax")) {
        double value;

        wxString valueMinStr = node->GetAttribute("min");
        if (!valueMinStr.ToDouble(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        float min = (float) value;

        wxString valueMaxStr = node->GetAttribute("max");
        if (!valueMaxStr.ToDouble(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        float max = (float) value;

        wxString valueStepStr = node->GetAttribute("step", "1");
        if (!valueStepStr.ToDouble(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        float step = (float) value;

        vect = BuildVectorFloat(min, max, step);
    } else {
        wxLogVerbose(_("The method is not correctly defined for %s in the calibration parameters file."), nodeName);
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorFloat(valueStr);
    }

    return vect;
}

VectorDouble asFileParameters::GetVectorDouble(wxXmlNode *node)
{
    VectorDouble vect;
    wxString nodeName = node->GetName();
    wxString method = node->GetAttribute("method");
    if (method.IsEmpty()) {
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorDouble(valueStr);
    } else if (method.IsSameAs("fixed")) {
        double value;
        wxString valueStr = node->GetChildren()->GetContent();
        if (!valueStr.ToDouble(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        vect.push_back(value);
    } else if (method.IsSameAs("array")) {
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorDouble(valueStr);
    } else if (method.IsSameAs("minmax")) {
        double min, max, step;

        wxString valueMinStr = node->GetAttribute("min");
        if (!valueMinStr.ToDouble(&min)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }

        wxString valueMaxStr = node->GetAttribute("max");
        if (!valueMaxStr.ToDouble(&max)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }

        wxString valueStepStr = node->GetAttribute("step", "1");
        if (!valueStepStr.ToDouble(&step)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }

        vect = BuildVectorDouble(min, max, step);
    } else {
        wxLogVerbose(_("The method is not correctly defined for %s in the calibration parameters file."), nodeName);
        wxString valueStr = node->GetChildren()->GetContent();
        vect = BuildVectorDouble(valueStr);
    }

    return vect;
}

VectorString asFileParameters::GetVectorString(wxXmlNode *node)
{
    VectorString vect;
    wxString nodeName = node->GetName();
    wxString method = node->GetAttribute("method");
    if (method.IsEmpty()) {
        wxString value = node->GetChildren()->GetContent();
        vect = BuildVectorString(value);
    } else if (method.IsSameAs("fixed")) {
        wxString value = node->GetChildren()->GetContent();
        value = value.Trim();
        value = value.Trim(true);
        vect.push_back(value);
    } else if (method.IsSameAs("array")) {
        wxString value = node->GetChildren()->GetContent();
        vect = BuildVectorString(value);
    } else {
        wxLogVerbose(_("The method is not correctly defined for %s in the calibration parameters file."), nodeName);
        wxString value = node->GetChildren()->GetContent();
        vect = BuildVectorString(value);
    }

    return vect;
}

VVectorInt asFileParameters::GetStationIdsVector(wxXmlNode *node)
{
    VVectorInt vect;
    wxString nodeName = node->GetName();
    wxString method = node->GetAttribute("method");
    if (method.IsEmpty()) {
        wxString value = node->GetChildren()->GetContent();
        VectorInt ids = GetStationIds(value);
        vect.push_back(ids);
    } else if (method.IsSameAs("fixed")) {
        wxString value = node->GetChildren()->GetContent();
        VectorInt ids = GetStationIds(value);
        vect.push_back(ids);
    } else if (method.IsSameAs("array")) {
        wxString value = node->GetChildren()->GetContent();

        // Explode the array
        wxChar separator = ',';
        while (value.Find(separator) != wxNOT_FOUND) {
            // If presence of a group next
            int startBracket = value.Find('(');
            if (startBracket != wxNOT_FOUND && startBracket < value.Find(separator)) {
                int endBracket = value.Find(')');
                wxString bracketContent = value.SubString((size_t) startBracket, (size_t) endBracket);
                VectorInt ids = GetStationIds(bracketContent);
                vect.push_back(ids);

                value = value.AfterFirst(')');
                if (value.Find(separator)) {
                    value = value.AfterFirst(separator);
                }
            } else {
                wxString txtbefore = value.BeforeFirst(separator);
                value = value.AfterFirst(separator);
                if (!txtbefore.IsEmpty()) {
                    VectorInt ids = GetStationIds(txtbefore);
                    vect.push_back(ids);
                }
            }
        }
        if (!value.IsEmpty()) {
            VectorInt ids = GetStationIds(value);
            vect.push_back(ids);
        }
    } else if (method.IsSameAs("minmax")) {
        long value;

        wxString valueMinStr = node->GetAttribute("min");
        if (!valueMinStr.ToLong(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        int min = (int) value;

        wxString valueMaxStr = node->GetAttribute("max");
        if (!valueMaxStr.ToLong(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        int max = (int) value;

        wxString valueStepStr = node->GetAttribute("step", "1");
        if (!valueStepStr.ToLong(&value)) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
        }
        int step = (int) value;

        VectorInt ids = BuildVectorInt(min, max, step);
        for (int i = 0; i < (int) ids.size(); i++) {
            VectorInt id;
            id.push_back(ids[i]);
            vect.push_back(id);
        }
    } else {
        wxLogVerbose(_("The method is not correctly defined for %s in the calibration parameters file."), nodeName);
        return VVectorInt(0);
    }

    return vect;
}

VectorInt asFileParameters::GetStationIds(wxString stationIdsString)
{
    // Trim
    stationIdsString.Trim(true);
    stationIdsString.Trim(false);

    VectorInt ids;

    if (stationIdsString.IsEmpty()) {
        wxLogError(_("The station ID was not provided."));
        return ids;
    }

    // Multivariate
    if (stationIdsString.SubString(0, 0).IsSameAs("(")) {
        wxString subStr = stationIdsString.SubString(1, stationIdsString.Len() - 1);

        // Check that it contains only 1 opening bracket
        if (subStr.Find("(") != wxNOT_FOUND) {
            wxLogError(_("The format of the station ID is not correct (more than one opening bracket)."));
            return ids;
        }

        // Check that it contains 1 closing bracket at the end
        if (subStr.Find(")") != (int) subStr.size() - 1) {
            wxLogError(_("The format of the station ID is not correct (location of the closing bracket)."));
            return ids;
        }

        // Extract content
        wxChar separator = ',';
        while (subStr.Find(separator) != wxNOT_FOUND) {
            wxString strBefore = subStr.BeforeFirst(separator);
            subStr = subStr.AfterFirst(separator);
            int id = wxAtoi(strBefore);
            ids.push_back(id);
        }
        if (!subStr.IsEmpty()) {
            int id = wxAtoi(subStr);
            ids.push_back(id);
        }
    } else {
        // Check for single value
        if (stationIdsString.Find("(") != wxNOT_FOUND || stationIdsString.Find(")") != wxNOT_FOUND ||
            stationIdsString.Find(",") != wxNOT_FOUND) {
            wxLogError(_("The format of the station ID is not correct (should be only digits)."));
            return ids;
        }
        int id = wxAtoi(stationIdsString);
        ids.push_back(id);
    }

    return ids;
}