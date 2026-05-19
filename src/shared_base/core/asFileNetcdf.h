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

#ifndef AS_FILE_NETCDF_H
#define AS_FILE_NETCDF_H

#include "asFile.h"
#include "asIncludes.h"
#include "netcdf.h"

/**
 * @brief NetCDF file class.
 *
 * This class is a wrapper around the NetCDF library.
 */
class asFileNetcdf : public asFile {
  public:
    enum Type {
        Byte,
        Char,
        Short,
        Int,
        Float,
        Double,
        String
    };

    enum Format {
        Classic = NC_FORMAT_CLASSIC,
        Format64bit = NC_FORMAT_64BIT,
        Netcdf4 = NC_FORMAT_NETCDF4,
        Netcdf4Classic = NC_FORMAT_NETCDF4_CLASSIC
    };

    asFileNetcdf(const wxString& fileName, const FileMode& fileMode);

    ~asFileNetcdf() override;

    [[nodiscard]] bool Open() override;

    [[nodiscard]] bool Close() override;

    [[nodiscard]] bool ForceClose();

    void DefDim(const wxString& dimName, const size_t& dimSize = 0);

    void DefVar(const wxString& varName, nc_type dataType, const int& varSize, const vstds& dimNames);

    void DefVarDeflate(const wxString& varName, int shuffle = 0, int deflateLevel = 2);

    void PutAtt(const wxString& attName, const wxString& textStr, const wxString& varName = wxEmptyString);

    void PutAtt(const wxString& attName, const short* attrValue, size_t length = 1,
                const wxString& varName = wxEmptyString);

    void PutAtt(const wxString& attName, const int* attrValue, size_t length = 1,
                const wxString& varName = wxEmptyString);

    void PutAtt(const wxString& attName, const float* attrValue, size_t length = 1,
                const wxString& varName = wxEmptyString);

    void PutAtt(const wxString& attName, const double* attrValue, size_t length = 1,
                const wxString& varName = wxEmptyString);

    void PutVarArray(const wxString& varName, const size_t* arrStart, const size_t* arrCount, const short* pData);

    void PutVarArray(const wxString& varName, const size_t* arrStart, const size_t* arrCount, const int* pData);

    void PutVarArray(const wxString& varName, const size_t* arrStart, const size_t* arrCount, const float* pData);

    void PutVarArray(const wxString& varName, const size_t* arrStart, const size_t* arrCount, const double* pData);

    void PutVarArray(const wxString& varName, const size_t* arrStart, const size_t* arrCount, const void* pData);

    void PutVarArray(const wxString& varName, const size_t* arrStart, const size_t* arrCount, const wxString* pData,
                     size_t totSize);

    void StartDef();

    void EndDef();

    [[nodiscard]] int GetDimId(const wxString& dimName);

    [[nodiscard]] bool HasVariable(const wxString& varName);

    [[nodiscard]] int GetVarId(const wxString& varName);

    [[nodiscard]] bool HasAttribute(const wxString& attName, const wxString& varName = wxEmptyString);

    [[nodiscard]] int GetAttId(const wxString& attName, const wxString& varName = wxEmptyString);

    [[nodiscard]] short GetAttShort(const wxString& attName, const wxString& varName = wxEmptyString);

    [[nodiscard]] int GetAttInt(const wxString& attName, const wxString& varName = wxEmptyString);

    [[nodiscard]] float GetAttFloat(const wxString& attName, const wxString& varName = wxEmptyString);

    [[nodiscard]] double GetAttDouble(const wxString& attName, const wxString& varName = wxEmptyString);

    [[nodiscard]] char GetAttChar(const wxString& attName, const wxString& varName = wxEmptyString);

    [[nodiscard]] wxString GetAttString(const wxString& attName, const wxString& varName = wxEmptyString);

    void GetVar(const wxString& varName, short* pValue);

    void GetVar(const wxString& varName, int* pValue);

    void GetVar(const wxString& varName, float* pValue);

    void GetVar(const wxString& varName, double* pValue);

    void GetVar(const wxString& varName, wxString* pValue, size_t totSize);

    [[nodiscard]] short GetVarOneShort(const wxString& varName, size_t arrIndex = 0);

    [[nodiscard]] int GetVarOneInt(const wxString& varName, size_t arrIndex = 0);

    [[nodiscard]] float GetVarOneFloat(const wxString& varName, size_t arrIndex = 0);

    [[nodiscard]] double GetVarOneDouble(const wxString& varName, size_t arrIndex = 0);

    void GetVarArray(const wxString& varName, const size_t indexStart[], const size_t indexCount[], short* pValue);

    void GetVarArray(const wxString& varName, const size_t indexStart[], const size_t indexCount[], int* pValue);

    void GetVarArray(const wxString& varName, const size_t indexStart[], const size_t indexCount[], float* pValue);

    void GetVarArray(const wxString& varName, const size_t indexStart[], const size_t indexCount[], double* pValue);

    void GetVarSample(const wxString& varName, const size_t indexStart[], const size_t indexCount[],
                      const ptrdiff_t indexStride[], short* pValue);

    void GetVarSample(const wxString& varName, const size_t indexStart[], const size_t indexCount[],
                      const ptrdiff_t indexStride[], int* pValue);

    void GetVarSample(const wxString& varName, const size_t indexStart[], const size_t indexCount[],
                      const ptrdiff_t indexStride[], float* pValue);

    void GetVarSample(const wxString& varName, const size_t indexStart[], const size_t indexCount[],
                      const ptrdiff_t indexStride[], double* pValue);

    [[nodiscard]] size_t GetDimLength(const wxString& dimName);

    [[nodiscard]] size_t GetVarLength(const wxString& varName);

    [[nodiscard]] nc_type GetVarType(const wxString& varName);

    [[nodiscard]] size_t GetVarsNb() const {
        return _struct.vars.size();
    }

    [[nodiscard]] size_t GetDimsNb() const {
        return _struct.dims.size();
    }

    [[nodiscard]] size_t GetGlobAttsNb() const {
        return _struct.atts.size();
    }

    [[nodiscard]] size_t GetVarAttsNb(int varId) const {
        return _struct.vars[varId].atts.size();
    }

    [[nodiscard]] size_t GetVarDimsNb(int varId) const {
        return _struct.vars[varId].dimIds.size();
    }

  protected:
  private:
    struct NcDimStruct {
        int id;
        wxString name;
        size_t length;
    };

    struct NcAttStruct {
        int id;
        wxString name;
        nc_type type;
        size_t length;
        void* pValue;
    };

    struct NcVarStruct {
        int id;
        wxString name;
        size_t length;
        nc_type type;
        vi dimIds;
        vector<NcAttStruct> atts;
    };

    struct NcStruct {
        int nUDims;
        vi uDimIds;
        asFileNetcdf::Format format;
        vector<NcDimStruct> dims;
        vector<NcVarStruct> vars;
        vector<NcAttStruct> atts;
    };

    NcStruct _struct;
    int _fileId;
    int _status;
    bool _defineMode;

    void HandleErrorNetcdf();

    void CheckDefModeOpen();

    void CheckDefModeClosed();

    void ClearStruct();

    bool ParseStruct();

    size_t GetVarLength(int& varId) const;
};

#endif
