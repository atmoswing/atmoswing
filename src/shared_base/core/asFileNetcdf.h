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

#ifndef ASFILENETCDF_H
#define ASFILENETCDF_H

#include "asIncludes.h"
#include <asFile.h>
#include "netcdf.h"


class asFileNetcdf
        : public asFile
{
public:
    enum Type
    {
        Byte, Char, Short, Int, Float, Double, String
    };

    enum Format
    {
        Classic = NC_FORMAT_CLASSIC,
        Format64bit = NC_FORMAT_64BIT,
        Netcdf4 = NC_FORMAT_NETCDF4,
        Netcdf4Classic = NC_FORMAT_NETCDF4_CLASSIC
    };

    asFileNetcdf(const wxString &FileName, const ListFileMode &FileMode);

    virtual ~asFileNetcdf();

    virtual bool Open();

    virtual bool Close();

    void DefDim(const wxString &DimName, const size_t &DimSize = 0);

    void DefVar(const wxString &VarName, nc_type DataType, const int &VarSize, const VectorStdString &DimNames);

    void DefVarDeflate(const wxString &VarName, int shuffle = 0, int deflateLevel = 2);

    void PutAtt(const wxString &AttName, const wxString &TextStr, const wxString &VarName = wxEmptyString);

    void PutAtt(const wxString &AttName, const short *attrValue, size_t Length = 1,
                const wxString &VarName = wxEmptyString);

    void PutAtt(const wxString &AttName, const int *attrValue, size_t Length = 1,
                const wxString &VarName = wxEmptyString);

    void PutAtt(const wxString &AttName, const float *attrValue, size_t Length = 1,
                const wxString &VarName = wxEmptyString);

    void PutAtt(const wxString &AttName, const double *attrValue, size_t Length = 1,
                const wxString &VarName = wxEmptyString);

    void PutVarArray(const wxString &VarName, const size_t *ArrStart, const size_t *ArrCount, const short *pData);

    void PutVarArray(const wxString &VarName, const size_t *ArrStart, const size_t *ArrCount, const int *pData);

    void PutVarArray(const wxString &VarName, const size_t *ArrStart, const size_t *ArrCount, const float *pData);

    void PutVarArray(const wxString &VarName, const size_t *ArrStart, const size_t *ArrCount, const double *pData);

    void PutVarArray(const wxString &VarName, const size_t *ArrStart, const size_t *ArrCount, const void *pData);

    void PutVarArray(const wxString &VarName, const size_t *ArrStart, const size_t *ArrCount, const wxString *pData,
                     const size_t TotSize);

    void StartDef();

    void EndDef();

    int GetDimId(const wxString &DimName);

    int GetVarId(const wxString &VarName);

    int GetAttId(const wxString &AttName, const wxString &VarName = wxEmptyString);

    short GetAttShort(const wxString &AttName, const wxString &VarName = wxEmptyString);

    int GetAttInt(const wxString &AttName, const wxString &VarName = wxEmptyString);

    float GetAttFloat(const wxString &AttName, const wxString &VarName = wxEmptyString);

    double GetAttDouble(const wxString &AttName, const wxString &VarName = wxEmptyString);

    char GetAttChar(const wxString &AttName, const wxString &VarName = wxEmptyString);

    wxString GetAttString(const wxString &AttName, const wxString &VarName = wxEmptyString);

    void GetVar(const wxString &VarName, short *pValue);

    void GetVar(const wxString &VarName, int *pValue);

    void GetVar(const wxString &VarName, float *pValue);

    void GetVar(const wxString &VarName, double *pValue);

    void GetVar(const wxString &VarName, wxString *pValue, const size_t TotSize);

    short GetVarOneShort(const wxString &VarName, size_t ArrIndex = 0);

    int GetVarOneInt(const wxString &VarName, size_t ArrIndex = 0);

    float GetVarOneFloat(const wxString &VarName, size_t ArrIndex = 0);

    double GetVarOneDouble(const wxString &VarName, size_t ArrIndex = 0);

    void GetVarArray(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], short *pValue);

    void GetVarArray(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], int *pValue);

    void GetVarArray(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], float *pValue);

    void GetVarArray(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], double *pValue);

    void GetVarSample(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[],
                      const ptrdiff_t IndexStride[], short *pValue);

    void GetVarSample(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[],
                      const ptrdiff_t IndexStride[], int *pValue);

    void GetVarSample(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[],
                      const ptrdiff_t IndexStride[], float *pValue);

    void GetVarSample(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[],
                      const ptrdiff_t IndexStride[], double *pValue);

    size_t GetDimLength(const wxString &DimName);

    size_t GetVarLength(const wxString &VarName);

    int GetFileId() const
    {
        return m_fileId;
    }

    int GetNVars() const
    {
        return m_struct.NVars;
    }

    int GetNDims() const
    {
        return m_struct.NDims;
    }

    int GetNGlobAtts() const
    {
        return m_struct.NAtts;
    }

protected:

private:
    struct ncDimStruct
    {
        int Id;
        wxString Name;
        size_t Length;
    };

    struct ncAttStruct
    {
        int Id;
        wxString Name;
        nc_type Type;
        size_t Length;
        void *pValue;
    };

    struct ncVarStruct
    {
        int Id;
        wxString Name;
        size_t Length;
        nc_type Type;
        int NDims;
        VectorInt NDimIds;
        int NAtts;
        std::vector<struct ncAttStruct> Atts;
    };

    struct ncStruct
    {
        int NDims;
        int NUDims;
        int NVars;
        int NAtts;
        int UDimId;
        VectorInt UDimsIds;
        asFileNetcdf::Format Format;
        std::vector<struct ncDimStruct> Dims;
        std::vector<struct ncVarStruct> Vars;
        std::vector<struct ncAttStruct> Atts;
    };

    int m_fileId;
    int m_status;
    bool m_defineMode;
    struct ncStruct m_struct;

    void HandleErrorNetcdf();

    void CheckDefModeOpen();

    void CheckDefModeClosed();

    void ClearStruct();

    bool ParseStruct();

    size_t GetVarLength(int &varid) const;

};

#endif // ASFILENETCDF_H
