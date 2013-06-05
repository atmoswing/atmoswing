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
 
#ifndef ASLOGGLOBALFUNCTIONS_H
#define ASLOGGLOBALFUNCTIONS_H

#include <asLog.h>
#include <asIncludes.h>

class asLog;

// Global log functions
extern asLog* g_pLog;
asLog& Log();
void DeleteLog();
void asLogError(const wxString &msg);
void asLogError(const wxChar* msg);
void asLogWarning(const wxString &msg);
void asLogWarning(const wxChar* msg);
void asLogMessage(const wxString &msg);
void asLogMessage(const wxChar* msg);
void asLogMessageImportant(const wxString &msg);
void asLogMessageImportant(const wxChar* msg);
void asLogState(const wxString &msg);
void asLogState(const wxChar* msg);
wxString asGetState();

#endif // ASLOGGLOBALFUNCTIONS_H
