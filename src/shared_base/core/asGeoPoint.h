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
 
#ifndef ASGEOPOINT_H
#define ASGEOPOINT_H

#include <asIncludes.h>
#include <asGeo.h>

class asGeoPoint: public asGeo
{
public:

    /** Default constructor
     * \param coosys The coordinate system
     * \param Point The coordinates of the point
     * \param Level The height in hPa
     * \param Level The height in m
     */
    asGeoPoint(CoordSys coosys, const Coo &Point, float Level = asNONE, float Height = asNONE);

    /** Other constructor
     * \param coosys The coordinate system
     * \param U The coordinate on the U axis
     * \param V The coordinate on the V axis
     * \param Level The height in hPa
     * \param Level The height in m
     */
    asGeoPoint(CoordSys coosys, double U, double V, float Level = asNONE, float Height = asNONE);

    /** Default destructor */
    virtual ~asGeoPoint();

    /** Access m_Point
     * \return The current value of m_Point
     */
    Coo GetCoo()
    {
        return m_Point;
    }

    /** Set m_Point
     * \param val New value to set
     */
    void SetCoo(const Coo &val)
    {
        m_Point = val;
        Init();
    }

    /** Gives the U coordinate
     * \return The coordinate on the U axis
     */
    double GetU()
    {
        return m_Point.u;
    }

    /** Gives the V coordinate
     * \return The coordinate on the V axis
     */
    double GetV()
    {
        return m_Point.v;
    }

    /** Access m_Level
     * \return The current value of m_Level
     */
    float GetLevel()
    {
        return m_Level;
    }

    /** Set m_Level
     * \param val New value to set
     */
    void SetLevel(float val)
    {
        m_Level = val;
    }

    /** Convert projection
     * \param newcoordsys The destination projection
     */
    void ProjConvert(CoordSys newcoordsys);

protected:
private:
    Coo m_Point; //!< Member variable "m_Point"
    float m_Level; //!< Member variable "m_Level" hPa
    float m_Height; //!< Member variable "m_Height" m

    /** Process to initialization and checks */
    void Init();

    /** Check every point
     * \return True on success
     */
    bool DoCheckPoints();
};

#endif
