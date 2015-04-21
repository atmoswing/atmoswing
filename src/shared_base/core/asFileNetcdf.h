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
 
#ifndef ASFILENETCDF_H
#define ASFILENETCDF_H

#include "asIncludes.h"
#include <asFile.h>
#include "netcdf.h"


class asFileNetcdf : public asFile
{
public:

    //!< The data types
    enum Type
    {
        Byte,
        Char,
        Short,
        Int,
        Float,
        Double,
        String
        /*
        Byte     = NC_BYTE,
        Char     = NC_CHAR,
        Short    = NC_SHORT,
        Int      = NC_INT,
        Float    = NC_FLOAT,
        Double   = NC_DOUBLE,
        String   = NC_STRING,
// TODO (phorton#5#): What about that ??
        UByte    = NC_UBYTE,
        UShort   = NC_USHORT,
        UInt     = NC_UINT,
        Int64    = NC_INT64,
        UInt64   = NC_UINT64,
        VLen     = NC_VLEN,
        Opaque   = NC_OPAQUE,
        Enum     = NC_ENUM,
        Compound = NC_COMPOUND*/
    };

    //!< The file format
    enum Format
    {
        Classic         = NC_FORMAT_CLASSIC,
        Format64bit     = NC_FORMAT_64BIT,
        Netcdf4         = NC_FORMAT_NETCDF4,
        Netcdf4Classic  = NC_FORMAT_NETCDF4_CLASSIC
    };

    /** Default constructor
     * \param FileName The full file path
     * \param FileMode The file access mode
     */
    asFileNetcdf(const wxString &FileName, const ListFileMode &FileMode);

    /** Default destructor */
    virtual ~asFileNetcdf();

    /** Open the file */
    virtual bool Open();

    /** Close the file */
    virtual bool Close();

    /** Add a dimension to the file
     * \param DimName The dimension name
     * \param DimSize The dimension size
     */
    void DefDim(const wxString &DimName, const size_t &DimSize = 0);

    /** Add a variable to the file
     * \param VarName The variable name
     * \param DataType The data types
     * \param VarSize The variable size
     * \param DimNames The corresponding dimensions
     */
    void DefVar(const wxString &VarName, nc_type DataType, const int &VarSize, const VectorStdString &DimNames);
    
    /** Add a variable to the file
     * \param VarName The variable name
     * \param shuffle If non-zero, turn on the shuffle filter
     * \param deflateLevel Deflate level (0-9)
     */
    void DefVarDeflate(const wxString &VarName, int shuffle = 0, int deflateLevel = 2);

    /** Set a text attribute value
     * \param AttName The attribute name
     * \param TextStr The text to set
     * \param VarName The variable name
     */
    void PutAtt(const wxString &AttName, const wxString &TextStr, const wxString &VarName = wxEmptyString);

    /** Set an attribute value of type short
     * \param AttName The attribute name
     * \param attrValue The value to set
     * \param Length The size of the array
     * \param VarName The variable name
     */
    void PutAtt(const wxString &AttName, const short* attrValue, size_t Length = 1, const wxString &VarName = wxEmptyString);

    /** Set an attribute value of type int
     * \param AttName The attribute name
     * \param attrValue The value to set
     * \param Length The size of the array
     * \param VarName The variable name
     */
    void PutAtt(const wxString &AttName, const int* attrValue, size_t Length = 1, const wxString &VarName = wxEmptyString);

    /** Set an attribute value of type float
     * \param AttName The attribute name
     * \param attrValue The value to set
     * \param Length The size of the array
     * \param VarName The variable name
     */
    void PutAtt(const wxString &AttName, const float* attrValue, size_t Length = 1, const wxString &VarName = wxEmptyString);

    /** Set an attribute value of type double
     * \param AttName The attribute name
     * \param attrValue The value to set
     * \param Length The size of the array
     * \param VarName The variable name
     */
    void PutAtt(const wxString &AttName, const double* attrValue, size_t Length = 1, const wxString &VarName = wxEmptyString);

    /** Write an array of shorts to the file
     * \param VarName The variable name
     * \param ArrStart The index of the begining of the data to write
     * \param ArrCount The number of elements of the data to write
     * \param pData A pointer to the data
     */
    void PutVarArray(const wxString &VarName, const size_t* ArrStart, const size_t* ArrCount, const short* pData);

    /** Write an array of ints to the file
     * \param VarName The variable name
     * \param ArrStart The index of the begining of the data to write
     * \param ArrCount The number of elements of the data to write
     * \param pData A pointer to the data
     */
    void PutVarArray(const wxString &VarName, const size_t* ArrStart, const size_t* ArrCount, const int* pData);

    /** Write an array of floats to the file
     * \param VarName The variable name
     * \param ArrStart The index of the begining of the data to write
     * \param ArrCount The number of elements of the data to write
     * \param pData A pointer to the data
     */
    void PutVarArray(const wxString &VarName, const size_t* ArrStart, const size_t* ArrCount, const float* pData);

    /** Write an array of doubles to the file
     * \param VarName The variable name
     * \param ArrStart The index of the begining of the data to write
     * \param ArrCount The number of elements of the data to write
     * \param pData A pointer to the data
     */
    void PutVarArray(const wxString &VarName, const size_t* ArrStart, const size_t* ArrCount, const double* pData);

    /** Write an array of values to the file
     * \param VarName The variable name
     * \param ArrStart The index of the begining of the data to write
     * \param ArrCount The number of elements of the data to write
     * \param pData A pointer to the data
     */
    void PutVarArray(const wxString &VarName, const size_t* ArrStart, const size_t* ArrCount, const void* pData);

    /** Write an array of string to the file
     * \param VarName The variable name
     * \param ArrStart The index of the begining of the data to write
     * \param ArrCount The number of elements of the data to write
     * \param pData A pointer to the data
     * \param TotSize The size of the original vector
     */
    void PutVarArray(const wxString &VarName, const size_t* ArrStart, const size_t* ArrCount, const wxString* pData, const size_t TotSize);

    /** Start the define mode */
    void StartDef();

    /** End the define mode */
    void EndDef();

    /** Get a dimension Id from its name
     * \param DimName The dimension name
     * \return The dimension Id
     */
    int GetDimId(const wxString &DimName);

    /** Get a variable Id from its name
     * \param VarName The variable name
     * \return The variable Id
     */
    int GetVarId(const wxString &VarName);

    /** Get an attribute Id from its name
     * \param AttName The attribute name
     * \param VarName The eventual variable name. If not set, search in global attributes
     * \return The variable Id
     */
    int GetAttId(const wxString &AttName, const wxString &VarName = wxEmptyString);

    /** Get an attribute value of type short
     * \param AttName The attribute name
     * \param VarName The eventual variable name. If not set, search in global attributes
     * \return The value
     */
    short GetAttShort(const wxString &AttName, const wxString &VarName = wxEmptyString);

    /** Get an attribute value of type int
     * \param AttName The attribute name
     * \param VarName The eventual variable name. If not set, search in global attributes
     * \return The value
     */
    int GetAttInt(const wxString &AttName, const wxString &VarName = wxEmptyString);

    /** Get an attribute value of type float
     * \param AttName The attribute name
     * \param VarName The eventual variable name. If not set, search in global attributes
     * \return The value
     */
    float GetAttFloat(const wxString &AttName, const wxString &VarName = wxEmptyString);

    /** Get an attribute value of type double
     * \param AttName The attribute name
     * \param VarName The eventual variable name. If not set, search in global attributes
     * \return The value
     */
    double GetAttDouble(const wxString &AttName, const wxString &VarName = wxEmptyString);

    /** Get an attribute value of type char
     * \param AttName The attribute name
     * \param VarName The eventual variable name. If not set, search in global attributes
     * \return The value
     */
    char GetAttChar(const wxString &AttName, const wxString &VarName = wxEmptyString);

    /** Get an attribute value of type wxString
     * \param AttName The attribute name
     * \param VarName The eventual variable name. If not set, search in global attributes
     * \return The value
     */
    wxString GetAttString(const wxString &AttName, const wxString &VarName = wxEmptyString);

    /** Get the whole variable values of type short
     * \param VarName The variable name
     * \return The value
     */
    void GetVar(const wxString &VarName, short* pValue);

    /** Get the whole variable values of type int
     * \param VarName The variable name
     * \return The value
     */
    void GetVar(const wxString &VarName, int* pValue);

    /** Get the whole variable values of type float
     * \param VarName The variable name
     * \return The value
     */
    void GetVar(const wxString &VarName, float* pValue);

    /** Get the whole variable values of type double
     * \param VarName The variable name
     * \return The value
     */
    void GetVar(const wxString &VarName, double* pValue);

    /** Get the whole variable values of type double
     * \param VarName The variable name
     * \param pValue The resulting container
     * \param TotSize The container size
     * \return
     */
    void GetVar(const wxString &VarName, wxString* pValue, const size_t TotSize);

    /** Get a unique variable value of type short
     * \param VarName The variable name
     * \param ArrIndex The index of the desired variable value
     * \return The value
     */
    short GetVarOneShort(const wxString &VarName, size_t ArrIndex = 0);

    /** Get a unique variable value of type int
     * \param VarName The variable name
     * \param ArrIndex The index of the desired variable value
     * \return The value
     */
    int GetVarOneInt(const wxString &VarName, size_t ArrIndex = 0);

    /** Get a unique variable value of type float
     * \param VarName The variable name
     * \param ArrIndex The index of the desired variable value
     * \return The value
     */
    float GetVarOneFloat(const wxString &VarName, size_t ArrIndex = 0);

    /** Get a unique variable value of type double
     * \param VarName The variable name
     * \param ArrIndex The index of the desired variable value
     * \return The value
     */
    double GetVarOneDouble(const wxString &VarName, size_t ArrIndex = 0);

    /** Get the full array of the variable value of type short
     * \param VarName The variable name
     * \param IndexStart The index of the begining of the desired variable values
     * \param IndexCount The index numbers of the desired variable values
     * \return The value
     */
    void GetVarArray(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], short* pValue);

    /** Get the full array of the variable value of type int
     * \param VarName The variable name
     * \param IndexStart The index of the begining of the desired variable values
     * \param IndexCount The index numbers of the desired variable values
     * \param NbDims The number of dimensions
     * \return The value
     */
    void GetVarArray(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], int* pValue);

    /** Get the full array of the variable value of type float
     * \param VarName The variable name
     * \param IndexStart The index of the begining of the desired variable values
     * \param IndexCount The index numbers of the desired variable values
     * \param NbDims The number of dimensions
     * \return The value
     */
    void GetVarArray(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], float* pValue);

    /** Get the full array of the variable value of type double
     * \param VarName The variable name
     * \param IndexStart The index of the begining of the desired variable values
     * \param IndexCount The index numbers of the desired variable values
     * \param NbDims The number of dimensions
     * \return The value
     */
    void GetVarArray(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], double* pValue);

    /** Get a subsampled array of the variable value of type short
     * \param VarName The variable name
     * \param IndexStart The index of the begining of the desired variable values
     * \param IndexCount The index numbers of the desired variable values
     * \param IndexStride The index strides of the desired variable values
     * \param NbDims The number of dimensions
     * \return The value
     */
    void GetVarSample(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], const ptrdiff_t IndexStride[], short* pValue);

    /** Get a subsampled array of the variable value of type int
     * \param VarName The variable name
     * \param IndexStart The index of the begining of the desired variable values
     * \param IndexCount The index numbers of the desired variable values
     * \param IndexStride The index strides of the desired variable values
     * \param NbDims The number of dimensions
     * \return The value
     */
    void GetVarSample(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], const ptrdiff_t IndexStride[], int* pValue);

    /** Get a subsampled array of the variable value of type float
     * \param VarName The variable name
     * \param IndexStart The index of the begining of the desired variable values
     * \param IndexCount The index numbers of the desired variable values
     * \param IndexStride The index strides of the desired variable values
     * \param NbDims The number of dimensions
     * \return The value
     */
    void GetVarSample(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], const ptrdiff_t IndexStride[], float* pValue);

    /** Get a subsampled array of the variable value of type double
     * \param VarName The variable name
     * \param IndexStart The index of the begining of the desired variable values
     * \param IndexCount The index numbers of the desired variable values
     * \param IndexStride The index strides of the desired variable values
     * \param NbDims The number of dimensions
     * \return The value
     */
    void GetVarSample(const wxString &VarName, const size_t IndexStart[], const size_t IndexCount[], const ptrdiff_t IndexStride[], double* pValue);

    /** Access the length of a dimension
     * \return The dimension length
     */
    size_t GetDimLength(const wxString &DimName);

    /** Access the length of a variable
     * \return The variable length
     */
    size_t GetVarLength(const wxString &VarName);

    /** Access m_fileId
     * \return The current value of m_fileId
     */
    int GetFileId()
    {
        return m_fileId;
    }

    /** Get the number of variables
     * \return The current value of m_struct.NVars
     */
    int GetNVars()
    {
        return m_struct.NVars;
    }

    /** Get the number of dimensions
     * \return The current value of m_struct.NDims
     */
    int GetNDims()
    {
        return m_struct.NDims;
    }

    /** Get the number of attributes
     * \return The current value of m_struct.NAtts
     */
    int GetNGlobAtts()
    {
        return m_struct.NAtts;
    }

protected:
private:
    //!< A structure for the NetCDF data properties
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
        void* pValue;
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

    int m_fileId; //!< Member variable "m_fileId"
    int m_status; //!< Member variable "m_status"
    bool m_defineMode; //!< Member variable "m_defineMode"
    struct ncStruct m_struct; //!< Member variable "m_struct"

    /** Method to handle NetCDF library errors */
    void HandleErrorNetcdf();

    /** Method to check if the file is in define mode */
    void CheckDefModeOpen();

    /** Method to check if the file is not in define mode */
    void CheckDefModeClosed();

    /** Method to clear the structure (avoid memory leaks) */
    void ClearStruct();

    /** Method to get the file properties */
    bool ParseStruct();

    /** Access the length of a variable
     * \return The variable length
     */
    size_t GetVarLength(int &varid);

};

#endif // ASFILENETCDF_H
